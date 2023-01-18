import json
import logging as log
import os
from typing import Tuple, Optional, Any

import numpy as np
import torch

from .data_loading import parallel_load_images
from .ray_utils import get_ray_directions, generate_hemispherical_orbit, get_rays
from .intrinsics import Intrinsics
from .base_dataset import BaseDataset


class SyntheticNerfDataset(BaseDataset):
    def __init__(self,
                 datadir,
                 split: str,
                 batch_size: Optional[int] = None,
                 downsample: float = 1.0,
                 max_frames: Optional[int] = None):
        self.downsample = downsample
        self.max_frames = max_frames
        self.near_far = [2.0, 6.0]

        if split == 'render':
            frames, transform = load_360_frames(datadir, 'test', self.max_frames)
            imgs, poses = load_360_images(frames, datadir, 'test', self.downsample)
            render_poses = generate_hemispherical_orbit(poses, n_frames=120)
            self.poses = render_poses
            intrinsics = load_360_intrinsics(
                transform, img_h=imgs[0].shape[0], img_w=imgs[0].shape[1],
                downsample=self.downsample)
            imgs = None
        else:
            frames, transform = load_360_frames(datadir, 'test', self.max_frames)
            imgs, poses = load_360_images(frames, datadir, 'test', self.downsample)
            intrinsics = load_360_intrinsics(
                transform, img_h=imgs[0].shape[0], img_w=imgs[0].shape[1],
                downsample=self.downsample)
        rays_o, rays_d, imgs = create_360_rays(
            imgs, poses, merge_all=split == 'train', intrinsics=intrinsics)
        super().__init__(
            datadir=datadir,
            split=split,
            scene_bbox=get_360_bbox(datadir, is_contracted=False),
            is_ndc=False,
            is_contracted=False,
            batch_size=batch_size,
            imgs=imgs,
            rays_o=rays_o,
            rays_d=rays_d,
            intrinsics=intrinsics,
        )
        log.info(f"SyntheticNerfDataset. Loaded {split} set from {datadir}."
                 f"{len(poses)} images of shape {self.img_h}x{self.img_w}. "
                 f"Images loaded: {imgs is not None}. "
                 f"Sampling without replacement={self.use_permutation}. {intrinsics}")

    def __getitem__(self, index):
        out = super().__getitem__(index)
        pixels = out["imgs"]
        if self.split == 'train':
            bg_color = torch.rand((1, 3), dtype=pixels.dtype, device=pixels.device)
        else:
            if pixels is None:
                bg_color = torch.ones((1, 3), dtype=torch.float32, device='cuda:0')
            else:
                bg_color = torch.ones((1, 3), dtype=pixels.dtype, device=pixels.device)
        # Alpha compositing
        if pixels is not None:
            pixels = pixels[:, :3] * pixels[:, 3:] + bg_color * (1.0 - pixels[:, 3:])
        out["imgs"] = pixels
        out["bg_color"] = bg_color
        out["near_fars"] = torch.tensor([[2.0, 6.0]])
        return out


def get_360_bbox(datadir, is_contracted=False):
    if is_contracted:
        radius = 2
    elif "ship" in datadir:
        radius = 1.5
    else:
        radius = 1.3
    return torch.tensor([[-radius, -radius, -radius], [radius, radius, radius]])


def create_360_rays(
              imgs: Optional[torch.Tensor],
              poses: torch.Tensor,
              merge_all: bool,
              intrinsics: Intrinsics) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    directions = get_ray_directions(intrinsics, opengl_camera=True)  # [H, W, 3]
    num_frames = poses.shape[0]

    all_rays_o, all_rays_d = [], []
    for i in range(num_frames):
        rays_o, rays_d = get_rays(directions, poses[i], ndc=False, normalize_rd=True)  # h*w, 3
        all_rays_o.append(rays_o)
        all_rays_d.append(rays_d)

    all_rays_o = torch.cat(all_rays_o, 0).to(dtype=torch.float32)  # [n_frames * h * w, 3]
    all_rays_d = torch.cat(all_rays_d, 0).to(dtype=torch.float32)  # [n_frames * h * w, 3]
    if imgs is not None:
        imgs = imgs.view(-1, imgs.shape[-1]).to(dtype=torch.float32)   # [N*H*W, 3/4]
    if not merge_all:
        num_pixels = intrinsics.height * intrinsics.width
        if imgs is not None:
            imgs = imgs.view(num_frames, num_pixels, -1)  # [N, H*W, 3/4]
        all_rays_o = all_rays_o.view(num_frames, num_pixels, 3)  # [N, H*W, 3]
        all_rays_d = all_rays_d.view(num_frames, num_pixels, 3)  # [N, H*W, 3]
    return all_rays_o, all_rays_d, imgs


def load_360_frames(datadir, split, max_frames: int) -> Tuple[Any, Any]:
    with open(os.path.join(datadir, f"transforms_{split}.json"), 'r') as f:
        meta = json.load(f)
        frames = meta['frames']

        # Subsample frames
        tot_frames = len(frames)
        num_frames = min(tot_frames, max_frames or tot_frames)
        if split == 'train' or split == 'test':
            subsample = int(round(tot_frames / num_frames))
            frame_ids = np.arange(tot_frames)[::subsample]
            if subsample > 1:
                log.info(f"Subsampling {split} set to 1 every {subsample} images.")
        else:
            frame_ids = np.arange(num_frames)
        frames = np.take(frames, frame_ids).tolist()
    return frames, meta


def load_360_images(frames, datadir, split, downsample) -> Tuple[torch.Tensor, torch.Tensor]:
    img_poses = parallel_load_images(
        dset_type="synthetic",
        tqdm_title=f'Loading {split} data',
        num_images=len(frames),
        frames=frames,
        data_dir=datadir,
        out_h=None,
        out_w=None,
        downsample=downsample,
    )
    imgs, poses = zip(*img_poses)
    imgs = torch.stack(imgs, 0)  # [N, H, W, 3/4]
    poses = torch.stack(poses, 0)  # [N, ????]
    return imgs, poses


def load_360_intrinsics(transform, img_h, img_w, downsample) -> Intrinsics:
    height = img_h
    width = img_w
    # load intrinsics
    if 'fl_x' in transform or 'fl_y' in transform:
        fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) / downsample
        fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) / downsample
    elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
        # blender, assert in radians. already downscaled since we use H/W
        fl_x = width / (2 * np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
        fl_y = height / (2 * np.tan(transform['camera_angle_y'] / 2)) if 'camera_angle_y' in transform else None
        if fl_x is None:
            fl_x = fl_y
        if fl_y is None:
            fl_y = fl_x
    else:
        raise RuntimeError('Failed to load focal length, please check the transforms.json!')

    cx = (transform['cx'] / downsample) if 'cx' in transform else (width / 2)
    cy = (transform['cy'] / downsample) if 'cy' in transform else (height / 2)
    return Intrinsics(height=height, width=width, focal_x=fl_x, focal_y=fl_y, center_x=cx, center_y=cy)
