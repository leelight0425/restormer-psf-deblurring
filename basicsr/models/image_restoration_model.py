import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')

import os
import random
import numpy as np
import cv2
import torch.nn.functional as F
from functools import partial

class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device

        self.use_identity = use_identity

        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1,1)).item()
    
        r_index = torch.randperm(target.size(0)).to(self.device)
    
        target = lam * target + (1-lam) * target[r_index, :]
        input_ = lam * input_ + (1-lam) * input_[r_index, :]
    
        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments)-1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_

class ImageCleanModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageCleanModel, self).__init__(opt)

        # define network

        self.mixing_flag = False
        if self.is_train:
            self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
            if self.mixing_flag:
                mixup_beta       = self.opt['train']['mixing_augs'].get('mixup_beta', 1.2)
                use_identity     = self.opt['train']['mixing_augs'].get('use_identity', False)
                self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity, self.device)

        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # The PSF encoder is owned by net_g, following the NAFNet design.
        self.kernel_channels = opt['network_g'].get('kernel_channels', 0)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

        # Load a legacy standalone PSF encoder, if explicitly configured.
        # New checkpoints include this module inside net_g.
        if self.kernel_channels > 0:
            psf_enc_path = self.opt['path'].get('pretrain_psf_encoder', None)
            if psf_enc_path is not None:
                logger = get_root_logger()
                logger.info(f'Loading PSF encoder from {psf_enc_path}')
                net_g = self.get_bare_model(self.net_g)
                self.load_network(net_g.psf_encoder, psf_enc_path,
                                  strict=True, param_key='params')

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(
                f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = define_network(self.opt['network_g']).to(
                self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path,
                                  self.opt['path'].get('strict_load_g',
                                                       True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            raise ValueError('pixel loss are None.')

        # gradient accumulation
        self.grad_accum_steps = train_opt.get('gradient_accumulation_steps', 1)
        if self.grad_accum_steps > 1:
            logger = get_root_logger()
            logger.info(
                f'Use Gradient Accumulation with {self.grad_accum_steps} steps. '
                f'Effective batch size: '
                f'{self.opt["datasets"]["train"].get("batch_size_per_gpu", 1) * self.grad_accum_steps}')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params, **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(optim_params, **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def _prepare_psf_kernel(self, data):
        """Prepare the PSF tensor for the encoder inside net_g."""
        self.kernel = None
        if self.kernel_channels > 0 and 'psf_kernel' in data:
            psf_k = data['psf_kernel'].to(self.device)  # (B, H, W, C) or (B, C, H, W)
            if psf_k.dim() == 4 and psf_k.shape[-1] in (1, 3):
                psf_k = psf_k.permute(0, 3, 1, 2)       # HWC -> CHW
            self.kernel = psf_k

    def feed_train_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        self._prepare_psf_kernel(data)

        if self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        self._prepare_psf_kernel(data)

    def optimize_parameters(self, current_iter):
        if self.grad_accum_steps <= 1:
            self.optimizer_g.zero_grad()

        preds = self.net_g(self.lq, kernel=self.kernel)
        if not isinstance(preds, list):
            preds = [preds]

        self.output = preds[-1]

        loss_dict = OrderedDict()
        # pixel loss (not scaled — logged at original magnitude)
        l_pix = 0.
        for pred in preds:
            l_pix += self.cri_pix(pred, self.gt)

        loss_dict['l_pix'] = l_pix

        # Scale loss for gradient accumulation
        (l_pix / self.grad_accum_steps).backward()

        if (current_iter + 1) % self.grad_accum_steps == 0:
            if self.opt['train']['use_grad_clip']:
                torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
            self.optimizer_g.step()
            self.optimizer_g.zero_grad()

            if self.ema_decay > 0:
                self.model_ema(decay=self.ema_decay)

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def pad_test(self, window_size):
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        self.nonpad_test(img)
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def nonpad_test(self, img=None):
        if img is None:
            img = self.lq
        kernel = self.kernel
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                pred = self.net_g_ema(img, kernel=kernel)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
        else:
            self.net_g.eval()
            with torch.no_grad():
                pred = self.net_g(img, kernel=kernel)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        pbar = tqdm(total=len(dataloader), unit='image', desc=f'{dataset_name}')

        window_size = self.opt['val'].get('window_size', 0)

        if window_size:
            test = partial(self.pad_test, window_size)
        else:
            test = self.nonpad_test

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            # log PSF info if available
            psf_cx = val_data.get('psf_cx', [None])[0]
            psf_cy = val_data.get('psf_cy', [None])[0]
            psf_info = f'psf=({psf_cx},{psf_cy})' if psf_cx is not None else ''

            self.feed_data(val_data)
            test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt

            # Save only the image channels. Gray+XY has three channels, but
            # its last two channels are coordinates rather than RGB.
            lq_tensor = visuals['lq']
            is_gray = visuals['result'].shape[1] == 1
            if is_gray:
                lq_tensor = lq_tensor[:, :1, :, :]
            elif lq_tensor.shape[1] > 3:
                lq_tensor = lq_tensor[:, :3, :, :]
            lq_img = tensor2img([lq_tensor], rgb2bgr=rgb2bgr)

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:

                if self.opt['is_train']:

                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}.png')

                    save_gt_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}_gt.png')
                    save_lq_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}_lq.png')
                else:
                    psf_tag = f'_psf({psf_cx},{psf_cy})' if psf_cx is not None else ''
                    save_img_path = osp.join(
                        self.opt['path']['visualization'], self.opt['name'],
                        f'{img_name}{psf_tag}.png')
                    save_gt_img_path = osp.join(
                        self.opt['path']['visualization'], self.opt['name'],
                        f'{img_name}{psf_tag}_gt.png')
                    save_lq_img_path = osp.join(
                        self.opt['path']['visualization'], self.opt['name'],
                        f'{img_name}{psf_tag}_lq.png')

                imwrite(sr_img, save_img_path)
                imwrite(gt_img, save_gt_img_path)
                imwrite(lq_img, save_lq_img_path)

            if with_metrics:
                # accumulate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        val = getattr(metric_module, metric_type)(sr_img, gt_img, **opt_)
                        self.metric_results[name] += val
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        val = getattr(metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)
                        self.metric_results[name] += val

            cnt += 1
            pbar.update(1)

        pbar.close()
        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)
        return current_metric


    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter, suffix=None):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema],
                              'net_g',
                              current_iter,
                              param_key=['params', 'params_ema'],
                              suffix=suffix)
        else:
            self.save_network(self.net_g, 'net_g', current_iter, suffix=suffix)
        if suffix is None:
            self.save_training_state(epoch, current_iter)
