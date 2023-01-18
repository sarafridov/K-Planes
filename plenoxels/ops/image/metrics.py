import math
import tempfile
import subprocess
import os
import re
from typing import List

import numpy as np
import skimage.metrics
import torch
from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure
import lpips

from .io import write_video_to_file, write_png


ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)

""" A module for image based metrics """


def psnr(rgb, gts):
    """Calculate the PSNR metric.

    Assumes the RGB image is in [0,1]

    Args:
        rgb (torch.Tensor): Image tensor of shape [H,W3]

    Returns:
        (float): The PSNR score
    """
    assert (rgb.max() <= 1.05 and rgb.min() >= -0.05)
    assert (gts.max() <= 1.05 and gts.min() >= -0.05)
    assert (rgb.shape[-1] == 3)
    assert (gts.shape[-1] == 3)

    mse = torch.mean((rgb[..., :3] - gts[..., :3]) ** 2).item()
    return 10 * math.log10(1.0 / mse)


def ssim(rgb, gts):
    """
    Modified from https://github.com/google/mipnerf/blob/16e73dfdb52044dcceb47cda5243a686391a6e0f/internal/math.py#L58
    """
    filter_size = 11
    filter_sigma = 1.5
    k1 = 0.01
    k2 = 0.03
    max_val = 1.0
    rgb = rgb.cpu().numpy()
    gts = gts.cpu().numpy()
    assert len(rgb.shape) == 3
    assert rgb.shape[-1] == 3
    assert rgb.shape == gts.shape
    import scipy.signal

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma)**2
    filt = np.exp(-0.5 * f_i)
    filt /= np.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    def convolve2d(z, f):
        return scipy.signal.convolve2d(z, f, mode='valid')

    filt_fn = lambda z: np.stack([
        convolve2d(convolve2d(z[..., i], filt[:, None]), filt[None, :])
        for i in range(z.shape[-1])], -1)
    mu0 = filt_fn(rgb)
    mu1 = filt_fn(gts)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(rgb**2) - mu00
    sigma11 = filt_fn(gts**2) - mu11
    sigma01 = filt_fn(rgb * gts) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = np.maximum(0., sigma00)
    sigma11 = np.maximum(0., sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(
        np.sqrt(sigma00 * sigma11), np.abs(sigma01))
    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    return np.mean(ssim_map)


def ssim_old(rgb, gts):
    """Calculate the SSIM metric.

    Assumes the RGB image is in [0,1]

    Args:
        rgb (torch.Tensor): Image tensor of shape [H,W,3]
        gts (torch.Tensor): Image tensor of shape [H,W,3]

    Returns:
        (float): The SSIM score
    """
    assert (rgb.max() <= 1.05 and rgb.min() >= -0.05)
    assert (gts.max() <= 1.05 and gts.min() >= -0.05)
    return skimage.metrics.structural_similarity(
        rgb[..., :3].cpu().numpy(),
        gts[..., :3].cpu().numpy(),
        channel_axis=2,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False)


def msssim(rgb, gts):
    assert (rgb.max() <= 1.05 and rgb.min() >= -0.05)
    assert (gts.max() <= 1.05 and gts.min() >= -0.05)
    return ms_ssim(torch.permute(rgb[None, ...], (0, 3, 1, 2)),
                   torch.permute(gts[None, ...], (0, 3, 1, 2))).item()


__LPIPS__ = {}


def init_lpips(net_name, device):
    return lpips.LPIPS(net=net_name, version='0.1').eval().to(device)


def rgb_lpips(rgb, gts, net_name='alex', device='cpu'):
    if net_name not in __LPIPS__:
        __LPIPS__[net_name] = init_lpips(net_name, device)
    gts = gts.permute([2, 0, 1]).contiguous().to(device)
    rgb = rgb.permute([2, 0, 1]).contiguous().to(device)
    return __LPIPS__[net_name](gts, rgb, normalize=True).item()


def jod(pred_frames: List[np.ndarray], gt_frames: List[np.ndarray]) -> float:
    with tempfile.TemporaryDirectory() as tmpdir:
        file_pred = os.path.join(tmpdir, "pred.mp4")
        write_video_to_file(file_pred, pred_frames)
        file_gt = os.path.join(tmpdir, "gt.mp4")
        write_video_to_file(file_gt, gt_frames)
        result = subprocess.check_output([
            'fvvdp', '--test', file_pred, '--ref', file_gt, '--gpu', '0',
            '--display', 'standard_fhd'])
        result = float(result.decode().strip().split('=')[1])
    return result


def flip(pred_frames: List[np.ndarray], gt_frames: List[np.ndarray], interval: int = 10) -> float:
    def extract_from_result(text: str, prompt: str):
        m = re.search(prompt, text)
        return float(m.group(1))

    all_results = []
    with tempfile.TemporaryDirectory() as tmpdir:
        pred_fname = os.path.join(tmpdir, "pred.png")
        gt_fname = os.path.join(tmpdir, "gt.png")
        for i in range(len(pred_frames)):
            if (i % interval) != 0:
                continue
            write_png(pred_fname, pred_frames[i])
            write_png(gt_fname, gt_frames[i])
            result = subprocess.check_output(
                ['python', 'plenoxels/ops/flip/flip.py', '--reference', gt_fname, '--test', pred_fname]
            ).decode()
            all_results.append(extract_from_result(result, r'Mean: (\d+\.\d+)'))
    return sum(all_results) / len(all_results)
