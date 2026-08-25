#!/usr/bin/env python
"""Run a Restormer PSF model on a real captured image.

The YAML file supplies the Restormer architecture. RGB models use global XY
coordinates for each tile, while models with ``kernel_channels > 0`` also
receive a PSF selected from the tile's sensor position.

Examples:
    python inference_psf.py \
        --opt PSF_Deblurring/Options/PSFDeblurringXY_NoEnc_Test.yml \
        --input photo.jpg --output restored.png --model model.pth

    python inference_psf.py \
        --opt PSF_Deblurring/Options/PSFDeblurring_Enc_Test.yml \
        --input photo.jpg --output restored.png \
        --model net_g_best.pth --psf_dir npz_07131
"""

import argparse
import copy
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicsr.models.archs import define_network
from basicsr.utils.options import parse


SENSOR_W = 4080
SENSOR_H = 3060
MODEL_INPUT_MULTIPLE = 8


def _as_existing_file(value):
    if value is None:
        return None
    path = Path(os.path.expanduser(str(value)))
    return path if path.is_file() else None


def _dataset_options(opt):
    datasets = opt.get('datasets', {})
    return (datasets.get('test') or datasets.get('val') or
            datasets.get('train') or {})


def _checkpoint_sort_key(path):
    stem = path.stem
    try:
        return int(stem.rsplit('_', 1)[1])
    except (IndexError, ValueError):
        return -1


def _find_model_checkpoint(opt):
    configured = _as_existing_file(
        opt.get('path', {}).get('pretrain_network_g'))
    if configured is not None:
        return configured

    roots = []
    configured_root = opt.get('path', {}).get('root')
    if configured_root:
        roots.append(Path(configured_root))
    roots.extend([ROOT, Path.cwd()])

    model_dirs = []
    for root in roots:
        model_dirs.append(root / 'experiments' / opt['name'] / 'models')

    seen = set()
    for model_dir in model_dirs:
        if str(model_dir) in seen or not model_dir.is_dir():
            continue
        seen.add(str(model_dir))
        for filename in ('net_g_best.pth', 'net_g_latest.pth'):
            candidate = model_dir / filename
            if candidate.is_file():
                return candidate
        checkpoints = list(model_dir.glob('net_g_*.pth'))
        if checkpoints:
            return max(checkpoints, key=_checkpoint_sort_key)

    return None


def _load_state_dict(path, device, param_key):
    try:
        checkpoint = torch.load(
            str(path), map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(str(path), map_location=device)

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get(param_key)
        if state_dict is None:
            fallback_key = 'params_ema' if param_key == 'params' else 'params'
            state_dict = checkpoint.get(fallback_key, checkpoint)
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError(f'Unsupported checkpoint format: {path}')

    return {
        key[7:] if key.startswith('module.') else key: value
        for key, value in state_dict.items()
    }


def _load_weights(module, path, device, param_key, strict):
    state_dict = _load_state_dict(path, device, param_key)
    incompatible = module.load_state_dict(state_dict, strict=strict)
    if (not strict and
            (incompatible.missing_keys or incompatible.unexpected_keys)):
        print(
            f'Warning: {path} loaded with '
            f'{len(incompatible.missing_keys)} missing and '
            f'{len(incompatible.unexpected_keys)} unexpected keys.')


def _resolve_mode(network_opt, requested):
    if requested == 'gray_psfenc':
        requested = 'gray'

    input_channels = int(network_opt.get('inp_channels', 3))
    if requested == 'auto':
        if input_channels == 1:
            requested = 'gray'
        elif input_channels == 3:
            requested = 'rgb'
        elif input_channels == 5:
            requested = 'rgb_xy'
        else:
            raise ValueError(
                'Cannot infer inference mode from '
                f'inp_channels={input_channels}.')

    expected_channels = {'gray': 1, 'rgb': 3, 'rgb_xy': 5}
    if requested not in expected_channels:
        raise ValueError(f'Unsupported inference mode: {requested}')
    if input_channels != expected_channels[requested]:
        raise ValueError(
            f'Mode {requested} requires inp_channels='
            f'{expected_channels[requested]}, got {input_channels}.')
    return requested


def _load_image(path, mode):
    flag = cv2.IMREAD_GRAYSCALE if mode == 'gray' else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), flag)
    if image is None:
        raise IOError(f'Failed to read input image: {path}')
    if mode == 'gray':
        return image.astype(np.float32) / 255.0
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image.astype(np.float32) / 255.0


def _load_psf(path, mode):
    with np.load(str(path)) as data:
        psf = np.asarray(data['psf'], dtype=np.float32)
        cx = int(data['cx']) if 'cx' in data.files else None
        cy = int(data['cy']) if 'cy' in data.files else None

    if psf.ndim == 2:
        psf = psf[:, :, None]
    if psf.ndim != 3:
        raise ValueError(f'PSF must be a 2D or 3D array: {path}')
    if mode == 'gray':
        psf = psf.mean(axis=2, keepdims=True)
    if psf.shape[0] != 31 or psf.shape[1] != 31:
        raise ValueError(f'Expected a 31x31 PSF kernel: {path}')

    return {'path': str(path), 'psf': psf, 'cx': cx, 'cy': cy}


def _load_psf_bank(directory, mode):
    files = sorted(Path(directory).glob('*.npz'))
    if not files:
        raise FileNotFoundError(f'No .npz files found in {directory}')
    bank = [_load_psf(path, mode) for path in files]
    if any(entry['cx'] is None or entry['cy'] is None for entry in bank):
        raise ValueError(
            'Every PSF in a PSF directory must contain cx and cy.')
    return bank


def _select_psf(bank, top, left, img_h, img_w, sensor_w, sensor_h):
    sensor_x = left / max(img_w - 1, 1) * (sensor_w - 1)
    sensor_y = top / max(img_h - 1, 1) * (sensor_h - 1)
    distances = [
        (entry['cx'] - sensor_x)**2 + (entry['cy'] - sensor_y)**2
        for entry in bank
    ]
    return bank[int(np.argmin(distances))]


def _make_xy_grid(h, w, top, left, img_h, img_w, sensor_w, sensor_h):
    step_x = sensor_w / max(img_w, 1)
    step_y = sensor_h / max(img_h, 1)
    sensor_x = left / max(img_w - 1, 1) * (sensor_w - 1)
    sensor_y = top / max(img_h - 1, 1) * (sensor_h - 1)
    xs = sensor_x + np.arange(w, dtype=np.float32) * step_x
    ys = sensor_y + np.arange(h, dtype=np.float32) * step_y
    xs = xs / (sensor_w - 1) * 2.0 - 1.0
    ys = ys / (sensor_h - 1) * 2.0 - 1.0
    xx, yy = np.meshgrid(xs, ys)
    return np.stack([xx, yy], axis=2)


def _pad_to_multiple(image, multiple=MODEL_INPUT_MULTIPLE):
    h, w = image.shape[:2]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if not pad_h and not pad_w:
        return image

    if image.ndim == 2:
        pad_width = ((0, pad_h), (0, pad_w))
    else:
        pad_width = ((0, pad_h), (0, pad_w), (0, 0))
    return np.pad(image, pad_width, mode='reflect')


def _process_tile(net, tile, top, left, img_h, img_w, device, mode,
                  psf_entry, sensor_w, sensor_h):
    tile_h, tile_w = tile.shape[:2]
    tile = _pad_to_multiple(tile)
    padded_h, padded_w = tile.shape[:2]

    if mode == 'gray':
        lq = tile[:, :, None]
    elif mode == 'rgb_xy':
        xy = _make_xy_grid(
            padded_h, padded_w, top, left, img_h, img_w, sensor_w, sensor_h)
        lq = np.concatenate([tile, xy], axis=2)
    else:
        lq = tile

    lq = np.ascontiguousarray(lq.transpose(2, 0, 1))
    lq_t = torch.from_numpy(lq).unsqueeze(0).to(device)

    kernel = None
    if psf_entry is not None:
        psf = psf_entry['psf'].transpose(2, 0, 1)
        kernel = torch.from_numpy(np.ascontiguousarray(psf)).float()
        kernel = kernel.unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = net(lq_t, kernel=kernel)
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[-1]

    prediction = prediction.squeeze(0).detach().cpu().numpy()
    prediction = prediction.transpose(1, 2, 0)
    prediction = np.clip(prediction, 0.0, 1.0)
    return prediction[:tile_h, :tile_w]


def _tile_starts(length, tile_size, stride):
    if length <= tile_size:
        return [0]
    last = length - tile_size
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return starts


def inference_full(net, image, device, mode, tile_size, tile_overlap,
                   psf_entry=None, psf_bank=None, sensor_w=SENSOR_W,
                   sensor_h=SENSOR_H):
    """Infer an image using overlapping tiles and global sensor coordinates."""
    if tile_size <= 0:
        raise ValueError('tile_size must be greater than zero.')
    if tile_overlap < 0 or tile_overlap >= tile_size:
        raise ValueError('tile_overlap must be in [0, tile_size).')

    img_h, img_w = image.shape[:2]
    out_channels = 1 if mode == 'gray' else 3
    stride = max(tile_size - tile_overlap, 1)
    result = np.zeros((img_h, img_w, out_channels), dtype=np.float32)
    count = np.zeros((img_h, img_w, 1), dtype=np.float32)

    for top in _tile_starts(img_h, tile_size, stride):
        for left in _tile_starts(img_w, tile_size, stride):
            tile = image[top:min(top + tile_size, img_h),
                         left:min(left + tile_size, img_w)]
            entry = psf_entry
            if entry is None and psf_bank is not None:
                entry = _select_psf(
                    psf_bank, top, left, img_h, img_w, sensor_w, sensor_h)
            prediction = _process_tile(
                net, tile, top, left, img_h, img_w, device,
                mode, entry, sensor_w, sensor_h)
            tile_h, tile_w = tile.shape[:2]
            result[top:top + tile_h, left:left + tile_w] += prediction
            count[top:top + tile_h, left:left + tile_w] += 1.0

    return result / np.maximum(count, 1.0)


def main():
    parser = argparse.ArgumentParser(
        description='Restormer inference for real captured images')
    parser.add_argument('--opt', required=True, help='Restormer YAML config')
    parser.add_argument('--input', required=True, help='Input photo path')
    parser.add_argument('--output', default='inference_result.png')
    parser.add_argument(
        '--model', default=None,
        help='net_g checkpoint; defaults to the configured or latest '
             'checkpoint')
    parser.add_argument('--psf', default=None, help='One fixed PSF .npz file')
    parser.add_argument(
        '--psf_dir', default=None,
        help='Directory of position-tagged PSF .npz files')
    parser.add_argument(
        '--mode', choices=['auto', 'rgb_xy', 'rgb', 'gray', 'gray_psfenc'],
        default='auto', help='Input mode; auto reads network_g.inp_channels')
    parser.add_argument('--tile_size', type=int, default=None)
    parser.add_argument('--tile_overlap', type=int, default=None)
    parser.add_argument('--sensor_width', type=int, default=None)
    parser.add_argument('--sensor_height', type=int, default=None)
    parser.add_argument(
        '--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument(
        '--param_key', choices=['params', 'params_ema'], default='params')
    args = parser.parse_args()

    if args.psf is not None and args.psf_dir is not None:
        parser.error('use either --psf or --psf_dir, not both')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        parser.error('CUDA was requested but is not available.')

    opt = parse(args.opt, is_train=False)
    network_opt = opt['network_g']
    try:
        mode = _resolve_mode(network_opt, args.mode)
    except ValueError as exc:
        parser.error(str(exc))
    expected_output_channels = 1 if mode == 'gray' else 3
    output_channels = int(network_opt.get('out_channels', 3))
    if output_channels != expected_output_channels:
        parser.error(
            f'Mode {mode} requires out_channels={expected_output_channels}, '
            f'got {output_channels}.')

    device = torch.device(args.device)
    if args.model:
        model_path = _as_existing_file(args.model)
        if model_path is None:
            parser.error(f'Model checkpoint does not exist: {args.model}')
    else:
        model_path = _find_model_checkpoint(opt)
    if model_path is None:
        parser.error('No Restormer checkpoint found; provide --model.')

    strict = opt.get('path', {}).get('strict_load_g', True)
    net = define_network(copy.deepcopy(network_opt)).to(device)
    _load_weights(net, model_path, device, args.param_key, strict)
    net.eval()
    print(f'Model loaded: {model_path}')

    kernel_channels = int(network_opt.get('kernel_channels', 0))

    dataset_opt = _dataset_options(opt)
    sensor_w = args.sensor_width or int(
        dataset_opt.get(
            'sensor_width', dataset_opt.get('sensor_w', SENSOR_W)))
    sensor_h = args.sensor_height or int(
        dataset_opt.get(
            'sensor_height', dataset_opt.get('sensor_h', SENSOR_H)))

    fixed_psf = None
    psf_bank = None
    psf_dir = args.psf_dir or dataset_opt.get('npz_dir') or dataset_opt.get(
        'psf_dir')
    if args.psf:
        try:
            fixed_psf = _load_psf(Path(args.psf), mode)
        except (FileNotFoundError, ValueError, KeyError) as exc:
            parser.error(str(exc))
    elif psf_dir and kernel_channels > 0:
        try:
            psf_bank = _load_psf_bank(Path(psf_dir), mode)
        except (FileNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        print(f'Loaded PSF bank: {len(psf_bank)} kernels')
    elif kernel_channels > 0:
        parser.error(
            'kernel_channels > 0 requires --psf, --psf_dir, or npz_dir '
            'in the YAML.')

    image = _load_image(Path(args.input), mode)
    tile_size = args.tile_size or int(dataset_opt.get('gt_size', 256))
    tile_overlap = (args.tile_overlap if args.tile_overlap is not None
                    else max(tile_size // 4, 0))
    print(
        f'Input: {image.shape[1]}x{image.shape[0]} | mode={mode} | '
        f'tile={tile_size} overlap={tile_overlap} | device={device}')

    result = inference_full(
        net, image, device, mode, tile_size, tile_overlap,
        psf_entry=fixed_psf, psf_bank=psf_bank,
        sensor_w=sensor_w, sensor_h=sensor_h)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_uint8 = np.round(result * 255.0).clip(0, 255).astype(np.uint8)
    if mode == 'gray':
        cv2.imwrite(str(output_path), result_uint8[:, :, 0])
    else:
        cv2.imwrite(
            str(output_path), cv2.cvtColor(result_uint8, cv2.COLOR_RGB2BGR))
    print(f'Result saved: {output_path}')


if __name__ == '__main__':
    main()
