#!/usr/bin/env python
"""Batch evaluation: evaluate every checkpoint, log PSNR/SSIM to tensorboard.

Usage:
    python PSF_Deblurring/batch_eval.py \
        --opt PSF_Deblurring/Options/PSFDeblurringXY_Test.yml \
        --models_dir E:/model/PSFDeblurringXY_Restormer_PE/models \
        --every 4

    tensorboard --logdir tb_logger/{name}_eval
"""

import argparse
import logging
import os
import sys
from collections import OrderedDict
from copy import deepcopy
from os import path as osp

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basicsr.data import create_dataloader, create_dataset
from basicsr.metrics import calculate_psnr, calculate_ssim
from basicsr.models import create_model
from basicsr.utils import (get_env_info, get_root_logger, get_time_str,
                           init_tb_logger, make_exp_dirs, tensor2img)
from basicsr.utils.dist_util import get_dist_info, init_dist
from basicsr.utils.options import dict2str, parse, ordered_yaml


def evaluate_one(model, val_loader, opt, current_iter, tb_logger, tag=''):
    """Run validation for one checkpoint."""
    prefix = f'{tag}/' if tag else ''
    with_metrics = opt['val'].get('metrics') is not None
    metric_results = {m: 0 for m in opt['val']['metrics'].keys()} if with_metrics else {}

    pbar = tqdm(total=len(val_loader), unit='img', desc=f'{prefix}iter={current_iter}', leave=False)

    for val_data in val_loader:
        model.feed_data(val_data)
        ws = opt['val'].get('window_size', 0)
        if ws:
            _, _, h0, w0 = model.lq.shape
            ph, pw = (ws - h0 % ws) % ws, (ws - w0 % ws) % ws
            if ph or pw:
                model.lq = torch.nn.functional.pad(model.lq, (0, pw, 0, ph), mode='reflect')
            model.nonpad_test()
            model.output = model.output[:, :, :h0, :w0]
        else:
            model.nonpad_test()

        visuals = model.get_current_visuals()
        rgb2bgr = opt['val'].get('rgb2bgr', True)
        sr = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)

        if with_metrics:
            use_img = opt['val'].get('use_image', True)
            if use_img and 'gt' in visuals:
                gt = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                for name, o_ in deepcopy(opt['val']['metrics']).items():
                    t_ = o_.pop('type')
                    fn = calculate_psnr if 'psnr' in name else calculate_ssim
                    metric_results[name] += fn(sr, gt, **o_)

        del model.lq, model.output
        if hasattr(model, 'gt'):
            del model.gt
        torch.cuda.empty_cache()
        pbar.update(1)
    pbar.close()

    cnt = max(len(val_loader), 1)
    for k in metric_results:
        metric_results[k] /= cnt

    if tb_logger is not None and with_metrics:
        for name, val in metric_results.items():
            tb_logger.add_scalar(f'{prefix}metrics/{name}', val, current_iter)

    return metric_results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--opt', type=str, default=r"PSF_Deblurring\Options\PSFDeblurring_Test.yml", help='Test YAML config')
    p.add_argument('--models_dir', type=str, default=r"E:\model\PSFDeblurringXY_Restormer_NonePE\models", help='Dir with .pth files')
    p.add_argument('--every', type=int, default=1, help='Eval every N ckpts')
    p.add_argument('--tag', type=str, default=None,
                   help='Label for this run (auto: models_dir basename)')
    args = p.parse_args()

    # Parse config
    opt = parse(args.opt, is_train=False)
    opt['dist'] = False
    opt['rank'], opt['world_size'] = get_dist_info()

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    if args.tag is None:
        args.tag = osp.basename(osp.normpath(args.models_dir))

    # Log
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'],
                        f"batch_eval_{opt['name']}_{args.tag}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO,
                              log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    logger.info(f'Models dir: {args.models_dir}')
    logger.info(f'Tag: {args.tag}')

    tb_dir = osp.join('tb_logger', f"{opt['name']}_eval")
    os.makedirs(tb_dir, exist_ok=True)
    tb_logger = init_tb_logger(log_dir=tb_dir)
    logger.info(f'Tensorboard: {osp.abspath(tb_dir)}')

    # Dataset
    ds_opt = opt['datasets']['test']
    ds_opt['phase'] = 'test'
    test_set = create_dataset(ds_opt)
    test_loader = create_dataloader(
        test_set, ds_opt, num_gpu=opt['num_gpu'], dist=False,
        sampler=None, seed=opt.get('manual_seed', 0))
    logger.info(f'Test images: {len(test_set)}')

    # Scan checkpoints (sorted by iter number, not lexicographic)
    def _ckpt_iter(fname):
        try:
            return int(osp.splitext(fname)[0].split('_')[-1])
        except ValueError:
            return 0

    all_pth = sorted(
        (f for f in os.listdir(args.models_dir)
         if f.startswith('net_g_') and f.endswith('.pth') and f != 'net_g_latest.pth'),
        key=_ckpt_iter)
    ckpts = [all_pth[i] for i in range(0, len(all_pth), args.every)]
    logger.info(f'{len(ckpts)} checkpoints (every={args.every})')

    # Create model once
    model = create_model(opt)
    model.net_g.eval()

    results = []
    for pth_name in tqdm(ckpts, desc='Eval'):
        try:
            it = int(os.path.splitext(pth_name)[0].split('_')[-1])
        except ValueError:
            it = 0

        ckpt_path = os.path.join(args.models_dir, pth_name)
        sd_raw = torch.load(ckpt_path, map_location=model.device)
        if 'params' in sd_raw:
            sd_raw = sd_raw['params']
        sd_raw = {k[7:] if k.startswith('module.') else k: v for k, v in sd_raw.items()}
        cur_sd = model.net_g.state_dict()
        sd = {k: v for k, v in sd_raw.items() if k in cur_sd and v.shape == cur_sd[k].shape}
        model.net_g.load_state_dict(sd, strict=False)

        m = evaluate_one(model, test_loader, opt, it, tb_logger, tag=args.tag)
        psnr, ssim = m.get('psnr', 0), m.get('ssim', 0)
        results.append((it, psnr, ssim))
        logger.info(f'  iter={it:>6d}  PSNR={psnr:.4f}  SSIM={ssim:.4f}')

    tb_logger.close()

    # Summary
    results.sort(key=lambda x: x[0])
    logger.info('=' * 60)
    logger.info(f'Done: {len(results)} checkpoints')
    best = max(results, key=lambda x: x[1])
    logger.info(f'Best PSNR: {best[1]:.4f} at iter={best[0]}')

    txt_path = osp.join(opt['path']['log'], f'batch_eval_{args.tag}_results.txt')
    with open(txt_path, 'w') as f:
        f.write('iter\tPSNR\tSSIM\n')
        for it, psnr, ssim in results:
            f.write(f'{it}\t{psnr:.6f}\t{ssim:.6f}\n')
    logger.info(f'Summary: {txt_path}')
    logger.info(f'Tensorboard: {osp.abspath(tb_dir)}')
    logger.info('=' * 60)


if __name__ == '__main__':
    main()
