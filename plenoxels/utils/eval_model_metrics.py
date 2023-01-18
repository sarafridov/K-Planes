"""Evaluate several metrics for a pretrained model. Handles video and static."""
import re
import glob
import os
from collections import defaultdict

import numpy as np
import torch

from plenoxels.ops.image import metrics
from plenoxels.ops.image.io import read_mp4, read_png


def eval_static_metrics(static_dir):
    all_file_names = glob.glob(os.path.join(static_dir, r"*.png"))
    # Collect all GT+Pred files per step
    files_per_step = defaultdict(list)
    for f in all_file_names:
        if "depth" in f:
            continue
        match = re.match(r".*step([0-9]+)-([0-9])+\.png", f)
        if match is None:
            continue
        step = int(match.group(1))
        files_per_step[step].append(f)
    steps = list(files_per_step.keys())
    max_step = max(steps)
    print(f"Evaluating static metrics for {static_dir} at step "
          f"{max_step} with {len(files_per_step[max_step])} files.")
    frames = [read_png(f) for f in files_per_step[max_step]]
    h1 = frames[0].shape[0] // 3
    pred_frames = [f[:h1] for f in frames]
    gt_frames = [f[h1:2*h1] for f in frames]

    psnrs, ssims, msssims, lpipss = [], [], [], []
    for pred, gt in zip(pred_frames, gt_frames):
        psnrs.append(metrics.psnr(pred, gt))
        ssims.append(metrics.ssim(pred, gt))
        msssims.append(metrics.msssim(pred, gt))
        lpipss.append(metrics.rgb_lpips(pred, gt, net_name="alex"))
    psnr = np.mean(psnrs)
    ssim = np.mean(ssims)
    msssim = np.mean(msssims)
    lpips = np.mean(lpipss)

    print()
    print(f"Images at {static_dir} - step {max_step}. Metrics:")
    print(f"PSNR = {psnr}")
    print(f"SSIM = {ssim}")
    print(f"MS-SSIM = {msssim}")
    print(f"Alex-LPIPS= {lpips}")
    print()


def eval_video_metrics(video_path):
    frames = read_mp4(video_path)
    h1 = frames[0].shape[0] // 3
    pred_frames = [f[:h1].numpy() for f in frames]
    gt_frames = [f[h1:2*h1].numpy() for f in frames]

    psnrs, ssims, msssims, lpipss = [], [], [], []
    for pred, gt in zip(pred_frames, gt_frames):
        pred = torch.from_numpy(pred).float().div(255)
        gt = torch.from_numpy(gt).float().div(255)
        psnrs.append(metrics.psnr(pred, gt))
        ssims.append(metrics.ssim(pred, gt))
        msssims.append(metrics.msssim(pred, gt))
        lpipss.append(metrics.rgb_lpips(pred, gt, net_name="alex"))
    psnr = np.mean(psnrs)
    ssim = np.mean(ssims)
    msssim = np.mean(msssims)
    lpips = np.mean(lpipss)

    flip = metrics.flip(pred_frames=pred_frames, gt_frames=gt_frames, interval=10)
    # jod = metrics.jod(pred_frames=pred_frames, gt_frames=gt_frames)

    print()
    print(f"Video at {video_path} metrics:")
    print(f"PSNR = {psnr}")
    print(f"SSIM = {ssim}")
    # print(f"MS-SSIM = {msssim}")
    # print(f"Alex-LPIPS = {lpips}")
    # print(f"FLIP = {flip}")
    # print(f"JOD = {jod}")
    print()
    print()


dnerf_scenes = ['hellwarrior', 'mutant', 'hook', 'bouncingballs', 'lego', 'trex', 'standup', 'jumpingjacks']
types = ['linear', 'mlp']

if __name__ == "__main__":
    for modeltype in types:
        for scene in dnerf_scenes:
            eval_video_metrics(f"logs/dnerf_{modeltype}_refactor1/{scene}_concat32_lr0.01_time0.1_tv0.0001_proptime0.001_proptv0.0001_distort0/step30000.mp4")
