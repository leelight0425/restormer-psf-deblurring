"""PSF Deblurring dataset with on-the-fly degradation.

Loads clean DIV2K images (from LMDB or disk), applies geometric augmentation
before randomly cropping patches, and applies spatially-varying PSF convolution
to create blurry LQ images.

PSF kernel selection: maps the crop offset to sensor coordinates and
selects the nearest PSF kernel by Euclidean distance.

Augmentation runs before degradation so the selected PSF, the XY channels,
and the generated blur all describe the same augmented sensor position.
"""

import random
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import convolve
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paths_from_lmdb
from basicsr.utils import FileClient, imfrombytes, img2tensor


class Dataset_PSFDeblurring(data.Dataset):
    """PSF Deblurring dataset with online spatially-varying PSF degradation.

    Supports two modes:
    1. LMDB (io_backend.type == 'lmdb'): reads clean GT keys from LMDB.
    2. Disk (io_backend.type == 'disk'): reads PNG/JPG from folder.

    Args:
        opt (dict): Config dict containing:
            dataroot_gt (str): Path to clean GT images (LMDB or folder).
            npz_dir (str): Directory with .npz PSF kernel files.
            use_flip (bool): Horizontal flip augmentation.
            use_rot (bool): Rotation augmentation (vertical flip + rot90).
            gt_size (int): Cropped patch size.
            io_backend (dict): IO backend config.
            phase (str): 'train' or 'val'.
            mean, std (tuple): Normalization params (optional).
            psf_edge_threshold (int): Edge PSF threshold for validation.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.backend_type = self.io_backend_opt['type']
        self.mean = opt.get('mean', None)
        self.std = opt.get('std', None)

        self.gt_folder = opt['dataroot_gt']
        self.phase = opt.get('phase', 'train')
        self.is_train = (self.phase == 'train')

        if self.backend_type == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.gt_folder]
            self.io_backend_opt['client_keys'] = ['gt']
            lmdb_keys = paths_from_lmdb(self.gt_folder)
            self.paths = [{'gt_path': k} for k in lmdb_keys]
        else:
            exts = {'.png', '.jpg', '.jpeg', '.bmp'}
            self.gt_paths = sorted(
                str(p) for p in Path(self.gt_folder).glob('*')
                if p.suffix.lower() in exts
            )
            if len(self.gt_paths) == 0:
                raise FileNotFoundError(f"No image files found in {self.gt_folder}")

        # ---- Load PSF kernels ----
        psf_dir = opt['npz_dir']
        npz_files = sorted(Path(psf_dir).glob('*.npz'))
        if len(npz_files) == 0:
            raise FileNotFoundError(f"No .npz files found in {psf_dir}")

        self.psf_kernels = []
        centers = []
        for f in npz_files:
            data_npz = np.load(str(f))
            self.psf_kernels.append(data_npz['psf'].astype(np.float32))
            centers.append((int(data_npz['cx']), int(data_npz['cy'])))
        self.psf_centers = np.array(centers, dtype=np.float32)

        # ---- Sensor dimensions ----
        self.sensor_w = 4080
        self.sensor_h = 3060

        # ---- Edge PSF filter (for validation) ----
        self.psf_edge_threshold = opt.get('psf_edge_threshold', 0)
        if self.psf_edge_threshold > 0 and not self.is_train:
            dist_left   = self.psf_centers[:, 0]
            dist_right  = self.sensor_w - self.psf_centers[:, 0]
            dist_top    = self.psf_centers[:, 1]
            dist_bottom = self.sensor_h - self.psf_centers[:, 1]
            edge_mask = (
                np.minimum(np.minimum(dist_left, dist_right),
                           np.minimum(dist_top, dist_bottom))
                < self.psf_edge_threshold
            )
            self.psf_edge_pool = np.where(edge_mask)[0].tolist()
            print(f'  [Dataset_PSFDeblurring] Edge PSF pool: {len(self.psf_edge_pool)} / '
                  f'{len(self.psf_centers)} (threshold={self.psf_edge_threshold})')
        else:
            self.psf_edge_pool = None

        # ---- Settings ----
        self.gt_size = opt.get('gt_size', 256)
        self.use_flip = opt.get('use_flip', True)
        self.use_rot = opt.get('use_rot', True)
        self.use_xy = opt.get('use_xy', False)

        xy_info = ' (XY)' if self.use_xy else ''
        n_paths = len(self.paths) if hasattr(self, 'paths') else len(self.gt_paths)
        print(f'  [Dataset_PSFDeblurring] {n_paths} images | {len(self.psf_kernels)} PSF kernels | '
              f'phase={self.phase} | gt_size={self.gt_size} | '
              f'sensor={self.sensor_w}x{self.sensor_h}{xy_info}')

    def __len__(self):
        if hasattr(self, 'paths'):
            return len(self.paths)
        return len(self.gt_paths)

    # ------------------------------------------------------------------
    # PSF utilities
    # ------------------------------------------------------------------

    def _map_offset_to_sensor(self, offset_x, offset_y, img_w, img_h):
        sensor_x = offset_x / max(img_w - 1, 1) * (self.sensor_w - 1)
        sensor_y = offset_y / max(img_h - 1, 1) * (self.sensor_h - 1)
        return sensor_x, sensor_y

    def _find_nearest_psf(self, offset_x, offset_y, img_w, img_h):
        sx, sy = self._map_offset_to_sensor(offset_x, offset_y, img_w, img_h)
        dists = np.sum((self.psf_centers - [sx, sy]) ** 2, axis=1)
        return int(np.argmin(dists))

    def _sample_edge_crop(self, img_w, img_h):
        """Sample a point in an edge strip, then select its nearest PSF."""
        max_top = img_h - self.gt_size
        max_left = img_w - self.gt_size
        max_sensor_x = self._map_offset_to_sensor(
            max_left, 0, img_w, img_h)[0]
        max_sensor_y = self._map_offset_to_sensor(
            0, max_top, img_w, img_h)[1]
        threshold = self.psf_edge_threshold

        side = random.choice(('left', 'right', 'top', 'bottom'))
        if side == 'left':
            sensor_x = random.uniform(0, min(threshold, max_sensor_x))
            sensor_y = random.uniform(0, max_sensor_y)
        elif side == 'right':
            sensor_x = random.uniform(
                max(0, max_sensor_x - threshold), max_sensor_x)
            sensor_y = random.uniform(0, max_sensor_y)
        elif side == 'top':
            sensor_x = random.uniform(0, max_sensor_x)
            sensor_y = random.uniform(0, min(threshold, max_sensor_y))
        else:
            sensor_x = random.uniform(0, max_sensor_x)
            sensor_y = random.uniform(
                max(0, max_sensor_y - threshold), max_sensor_y)

        left = int(round(sensor_x / max(self.sensor_w - 1, 1) *
                         max(img_w - 1, 1)))
        top = int(round(sensor_y / max(self.sensor_h - 1, 1) *
                        max(img_h - 1, 1)))
        left = int(np.clip(left, 0, max_left))
        top = int(np.clip(top, 0, max_top))
        psf_idx = self._find_nearest_psf(left, top, img_w, img_h)
        return top, left, psf_idx

    @staticmethod
    def _psf_convolve_rgb(image, psf):
        """Apply the reference dataset's per-channel PSF convolution."""
        image = np.ascontiguousarray(image, dtype=np.float64)
        psf = np.ascontiguousarray(psf, dtype=np.float64)
        convolved = np.zeros_like(image)
        for ch in range(3):
            convolved[:, :, ch] = convolve(
                image[:, :, ch], psf[:, :, ch], mode='reflect')
        return convolved.astype(np.float32)

    # ------------------------------------------------------------------
    # XY grid utilities
    # ------------------------------------------------------------------

    def _make_xy_grid(self, h, w, sensor_x, sensor_y, step_x, step_y):
        xs = sensor_x + np.arange(w, dtype=np.float32) * step_x
        ys = sensor_y + np.arange(h, dtype=np.float32) * step_y
        xs_norm = xs / (self.sensor_w - 1) * 2.0 - 1.0
        ys_norm = ys / (self.sensor_h - 1) * 2.0 - 1.0
        xx, yy = np.meshgrid(xs_norm, ys_norm)
        return np.stack([xx, yy], axis=2)  # (H, W, 2)

    # ------------------------------------------------------------------
    # Main __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # ---- 1. Load GT image ----
        if self.backend_type == 'lmdb':
            gt_path = self.paths[index % len(self.paths)]['gt_path']
            img_bytes = self.file_client.get(gt_path, 'gt')
            try:
                img_gt = imfrombytes(img_bytes, float32=True)
            except Exception:
                raise IOError(f"gt path {gt_path} not working")
            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
            gt_path_str = gt_path
        else:
            gt_path_str = self.gt_paths[index % len(self.gt_paths)]
            img_gt = cv2.imread(gt_path_str)
            if img_gt is None:
                raise IOError(f"Failed to read image: {gt_path_str}")
            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        h_img, w_img = img_gt.shape[:2]

        # ---- 2. Pad if smaller than gt_size ----
        h_pad = max(0, self.gt_size - h_img)
        w_pad = max(0, self.gt_size - w_img)
        if h_pad > 0 or w_pad > 0:
            img_gt = cv2.copyMakeBorder(
                img_gt, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
            h_img, w_img = img_gt.shape[:2]

        # ---- 3. Geometric augmentation BEFORE degradation ----
        # Select the PSF and generate XY coordinates from the augmented image
        # so they remain consistent with the blur applied to the crop.
        if self.is_train and (self.use_flip or self.use_rot):
            hflip = self.use_flip and random.random() < 0.5
            vflip = self.use_rot and random.random() < 0.5
            rot90 = self.use_rot and random.random() < 0.5
            if hflip:
                img_gt = img_gt[:, ::-1, :].copy()
            if vflip:
                img_gt = img_gt[::-1, :, :].copy()
            if rot90:
                img_gt = img_gt.transpose(1, 0, 2).copy()
            h_img, w_img = img_gt.shape[:2]

        # ---- 4. Crop + select PSF (in the augmented image) ----
        if self.is_train:
            top = random.randint(0, h_img - self.gt_size)
            left = random.randint(0, w_img - self.gt_size)
            psf_idx = self._find_nearest_psf(left, top, w_img, h_img)
            psf_xy_top, psf_xy_left = top, left
        elif self.psf_edge_pool is not None:
            top, left, psf_idx = self._sample_edge_crop(w_img, h_img)
            psf_xy_top, psf_xy_left = top, left
        else:
            top = (h_img - self.gt_size) // 2
            left = (w_img - self.gt_size) // 2
            rand_top = random.randint(0, h_img - self.gt_size)
            rand_left = random.randint(0, w_img - self.gt_size)
            psf_idx = self._find_nearest_psf(rand_lef   t, rand_top, w_img, h_img)
            psf_xy_top, psf_xy_left = rand_top, rand_left

        img_gt_crop = img_gt[top:top + self.gt_size, left:left + self.gt_size, :].copy()

        # ---- 5. PSF convolution: augmented GT -> LQ (RGB domain) ----
        psf = self.psf_kernels[psf_idx]
        img_lq = self._psf_convolve_rgb(img_gt_crop, psf)
        img_lq = np.clip(img_lq, 0.0, 1.0)

        # ---- 6. XY channel handling ----
        if self.use_xy:
            step_x = self.sensor_w / w_img
            step_y = self.sensor_h / h_img
            sensor_x, sensor_y = self._map_offset_to_sensor(
                psf_xy_left, psf_xy_top, w_img, h_img)

            xy_grid = self._make_xy_grid(self.gt_size, self.gt_size,
                                         sensor_x, sensor_y, step_x, step_y)

            # Stack LQ + XY -> (H, W, 3+2)
            img_lq = np.concatenate([img_lq, xy_grid], axis=2)

        # ---- 7. Convert to tensor (already RGB) ----
        img_gt_t = img2tensor([img_gt_crop], bgr2rgb=False, float32=True)[0]
        img_lq_t = img2tensor([img_lq], bgr2rgb=False, float32=True)[0]

        # ---- 8. Normalize (optional) ----
        if self.mean is not None and self.std is not None:
            normalize(img_lq_t, self.mean, self.std, inplace=True)
            normalize(img_gt_t, self.mean, self.std, inplace=True)

        psf_kernel = self.psf_kernels[psf_idx].copy()  # (31, 31, 3) float32

        return {
            'lq': img_lq_t,
            'gt': img_gt_t,
            'lq_path': gt_path_str,
            'gt_path': gt_path_str,
            'psf_cx': int(self.psf_centers[psf_idx][0]),
            'psf_cy': int(self.psf_centers[psf_idx][1]),
            'psf_kernel': psf_kernel,                  # (31, 31, 3) float32, for PSF encoder
        }
