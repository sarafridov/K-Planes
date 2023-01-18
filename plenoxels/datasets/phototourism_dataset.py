import glob
import math
import logging as log
import os
from enum import Enum
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import torch

from plenoxels.datasets.base_dataset import BaseDataset
from plenoxels.datasets.intrinsics import Intrinsics
from plenoxels.datasets.ray_utils import average_poses
from plenoxels.ops.image.io import read_png


class PhototourismScenes(Enum):
    TREVI = "trevi"
    BRANDENBURG = "brandenburg"
    SACRE = "sacre"

    @staticmethod
    def get_scene_from_datadir(datadir: str) -> 'PhototourismScenes':
        if "sacre" in datadir:
            return PhototourismScenes.SACRE
        if "trevi" in datadir:
            return PhototourismScenes.TREVI
        if "brandenburg" in datadir:
            return PhototourismScenes.BRANDENBURG
        raise NotImplementedError(datadir)


class PhotoTourismDataset(BaseDataset):
    def __init__(self,
                 datadir: str,
                 split: str,
                 batch_size: Optional[int] = None,
                 contraction: bool = False,
                 ndc: bool = False,
                 scene_bbox: Optional[List] = None,
                 global_translation: List[float] = None,
                 global_scale: List[float] = None,
                 downsample: float = 1.0):
        if ndc:
            raise NotImplementedError("PhotoTourism only handles contraction and standard.")
        if downsample != 1.0:
            raise NotImplementedError("PhotoTourism does not handle image downsampling.")
        if (global_scale is None or global_translation is None) and contraction:
            raise ValueError("scale and translation must be specified when contraction is used.")
        if not os.path.isdir(datadir):
            raise ValueError(f"Directory {datadir} does not exist.")

        if split == 'train' or split == 'test':
            pt_data_file = os.path.join(datadir, f"cache_{split}.pt")
            if not os.path.isfile(pt_data_file):
                # Populate cache
                cache_data(datadir=datadir, split=split, out_fname=os.path.basename(pt_data_file))
            pt_data = torch.load(pt_data_file)

            intrinsics = [
                Intrinsics(width=img.shape[1], height=img.shape[0],
                           center_x=img.shape[1] / 2, center_y=img.shape[0] / 2,
                           # focals are unused, we reuse intrinsics from Matt's files.
                           focal_y=0, focal_x=0)
                for img in pt_data["images"]
            ]
            if split == 'train':
                near_fars = torch.cat([
                    pt_data["bounds"][i].expand(intrinsics[i].width * intrinsics[i].height, 2)
                    for i in range(len(intrinsics))
                ], dim=0)
                camera_ids = torch.cat([
                    pt_data["camera_ids"][i].expand(intrinsics[i].width * intrinsics[i].height, 1)
                    for i in range(len(intrinsics))
                ])
                images = torch.cat([img.view(-1, 3) for img in pt_data["images"]], 0)
                rays_o = torch.cat([ro.view(-1, 3) for ro in pt_data["rays_o"]], 0)
                rays_d = torch.cat([rd.view(-1, 3) for rd in pt_data["rays_d"]], 0)
            elif split == 'test':
                images = pt_data["images"]
                rays_o = pt_data["rays_o"]
                rays_d = pt_data["rays_d"]
                near_fars = pt_data["bounds"]
                camera_ids = pt_data["camera_ids"]
        elif split == 'render':
            n_frames, frame_size = 150, 800
            rays_o, rays_d, camera_ids, near_fars = pt_render_poses(
                datadir, n_frames=n_frames, frame_h=frame_size, frame_w=frame_size)
            images = None
            intrinsics = [
                Intrinsics(width=frame_size, height=frame_size, focal_x=0, focal_y=0,
                           center_x=frame_size / 2, center_y=frame_size / 2)
                for _ in range(n_frames)
            ]
        else:
            raise NotImplementedError(split)

        # ugly hack: num_images needs to be the number of training images. This is needed to
        # initialize the appearance embedding tensor to the correct size even if we're just
        # reloading a previous model.
        self.num_images = self.get_num_train_images(datadir)
        self.camera_ids = camera_ids  # noqa
        self.near_fars = near_fars  # noqa

        self.global_scale, self.global_translation = None, None
        if contraction:
            self.global_translation = torch.as_tensor(global_translation).float()
            self.global_scale = torch.as_tensor(global_scale).float()

        if scene_bbox is None:
            raise ValueError("Must specify scene_bbox")
        scene_bbox = torch.tensor(scene_bbox)

        super().__init__(
            datadir=datadir,
            split=split,
            batch_size=batch_size,
            is_ndc=ndc,
            is_contracted=contraction,
            scene_bbox=scene_bbox,
            rays_o=rays_o,  # noqa
            rays_d=rays_d,  # noqa
            intrinsics=intrinsics,
            imgs=images,  # noqa
        )
        log.info(f"PhotoTourismDataset contracted={self.is_contracted}, ndc={self.is_ndc}. "
                 f"Loaded {self.split} set from {self.datadir}: "
                 f"{len(intrinsics)} images of sizes between {min(self.img_h)}x{min(self.img_w)} "
                 f"and {max(self.img_h)}x{max(self.img_w)}. "
                 f"Images loaded: {self.imgs is not None}.")
        if self.is_contracted:
            log.info(f"Contraction parameters: global_translation={self.global_translation}, "
                     f"global_scale={self.global_scale}")
        else:
            log.info(f"Bounding box: {self.scene_bbox}")

    def __getitem__(self, index):
        out, index = super().__getitem__(index, return_idxs=True)
        out["bg_color"] = torch.ones((1, 3), dtype=torch.float32)
        out["timestamps"] = self.camera_ids[index]
        out["near_fars"] = self.near_fars[index]
        if self.imgs is not None:
            out["imgs"] = out["imgs"] / 255.0  # this converts to f32

        if self.split != 'train':  # gen left-image and reshape correctly
            intrinsics = self.intrinsics[index]
            img_h, img_w = intrinsics.height, intrinsics.width
            mid = img_w // 2
            if self.imgs is not None:
                out["imgs_left"] = out["imgs"][:, :mid, :].reshape(-1, 3)
                out["rays_o_left"] = out["rays_o"].view(img_h, img_w, 3)[:, :mid, :].reshape(-1, 3)
                out["rays_d_left"] = out["rays_d"].view(img_h, img_w, 3)[:, :mid, :].reshape(-1, 3)
                out["imgs"] = out["imgs"].view(-1, 3)
            out["rays_o"] = out["rays_o"].reshape(-1, 3)
            out["rays_d"] = out["rays_d"].reshape(-1, 3)
            out["timestamps"] = out["timestamps"].expand(out["rays_o"].shape[0], 1)
            out["near_fars"] = out["near_fars"].expand(out["rays_o"].shape[0], 2)
        return out

    @staticmethod
    def get_num_train_images(datadir) -> int:
        scene = PhototourismScenes.get_scene_from_datadir(datadir)
        if scene == PhototourismScenes.TREVI:
            return 1689
        elif scene == PhototourismScenes.BRANDENBURG:
            return 763
        elif scene == PhototourismScenes.SACRE:
            return 830
        else:
            raise ValueError(scene)


def get_rays_tourism(H, W, kinv, pose):
    """
    phototourism camera intrinsics are defined by H, W and kinv.
    Args:
        H: image height
        W: image width
        kinv (3, 3): inverse of camera intrinsic
        pose (4, 4): camera extrinsic
    Returns:
        rays_o (H, W, 3): ray origins
        rays_d (H, W, 3): ray directions
    """
    yy, xx = torch.meshgrid(torch.arange(0., H, device=kinv.device),
                            torch.arange(0., W, device=kinv.device),
                            indexing='ij')
    pixco = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)

    directions = torch.matmul(pixco, kinv.T)  # (H, W, 3)

    rays_d = torch.matmul(directions, pose[:3, :3].T)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)  # (H, W, 3)

    rays_o = pose[:3, -1].expand_as(rays_d)  # (H, W, 3)

    return rays_o, rays_d


def load_camera_metadata(datadir: str, idx) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        poses = np.load(str(Path(datadir) / "c2w_mats.npy"))[idx]
        kinvs = np.load(str(Path(datadir) / "kinv_mats.npy"))[idx]
        bounds = np.load(str(Path(datadir) / "bds.npy"))[idx]
        res = np.load(str(Path(datadir) / "res_mats.npy"))[idx]
    except FileNotFoundError as e:
        error_msg = (
            f"One of the needed Phototourism files does not exist ({e.filename}). "
            f"They can be downloaded from "
            f"https://drive.google.com/drive/folders/1SVHKRQXiRb98q4KHVEbj8eoWxjNS2QLW"
        )
        log.error(error_msg)
        raise e
    return poses, kinvs, bounds, res


def get_ids_for_split(datadir, split):
    # read all files in the tsv first (split to train and test later)
    tsv = glob.glob(os.path.join(datadir, '*.tsv'))[0]
    files = pd.read_csv(tsv, sep='\t')
    files = files[~files['id'].isnull()]  # remove data without id
    files.reset_index(inplace=True, drop=True)
    files = files[files["split"] == split]

    imagepaths = sorted((Path(datadir) / "dense" / "images").glob("*.jpg"))
    imkey = np.array([os.path.basename(im) for im in imagepaths])
    idx = np.in1d(imkey, files["filename"])
    return idx, imagepaths


def scale_cam_metadata(poses: np.ndarray, kinvs: np.ndarray, bounds: np.ndarray, scale: float = 0.05):
    poses[:, :3, 3:4] *= scale
    bounds = bounds * scale * np.array([0.9, 1.2])

    return poses, kinvs, bounds


def cache_data(datadir: str, split: str, out_fname: str):
    log.info(f"Preparing cached rays for dataset at {datadir} - {split=}.")
    scale = 0.05

    idx, imagepaths = get_ids_for_split(datadir, split)

    imagepaths = np.array(imagepaths)[idx]
    poses, kinvs, bounds, res = load_camera_metadata(datadir, idx)
    poses, kinvs, bounds = scale_cam_metadata(poses, kinvs, bounds, scale=scale)
    img_w = res[:, 0]
    img_h = res[:, 1]
    size = int(np.sum(img_w * img_h))
    log.info(f"Loading dataset from {datadir}. Num images={len(imagepaths)}. Total rays={size}.")

    all_images, all_rays_o, all_rays_d, all_bounds, all_camera_ids = [], [], [], [], []
    for idx, impath in enumerate(imagepaths):
        image = read_png(impath)

        pose = torch.from_numpy(poses[idx]).float()
        kinv = torch.from_numpy(kinvs[idx]).float()
        bound = torch.from_numpy(bounds[idx]).float()

        rays_o, rays_d = get_rays_tourism(image.shape[0], image.shape[1], kinv, pose)

        camera_id = torch.tensor(idx)

        all_images.append(image.mul(255).to(torch.uint8))
        all_rays_o.append(rays_o)
        all_rays_d.append(rays_d)
        all_bounds.append(bound)
        all_camera_ids.append(camera_id)

    torch.save({
        "images": all_images,
        "rays_o": all_rays_o,
        "rays_d": all_rays_d,
        "bounds": all_bounds,
        "camera_ids": all_camera_ids,
    }, os.path.join(datadir, out_fname))


def pt_spiral_path(
        scene: PhototourismScenes,
        poses: torch.Tensor,
        n_frames=120,
        n_rots: float = 1.0,
        zrate=.5) -> torch.Tensor:
    if poses.shape[1] > 3:
        poses = poses[:, :3, :]
    c2w = torch.from_numpy(average_poses(poses.numpy()))  # [3, 4]

    # Generate poses for spiral path.
    render_poses = []
    for theta in np.linspace(0., 2. * np.pi * n_rots, n_frames, endpoint=False):
        rotation = c2w[:3, :3]
        # the additive translation vectors have 3 components (x, y, z axes)
        # each with an additive part - which defines a global shift of all poses
        # and a multiplicative part which changes the amplitude of movement
        # of the poses.
        if scene == PhototourismScenes.SACRE:
            translation = c2w[:, 3:4] + torch.tensor([[
                0.01 + 0.03 * np.cos(theta),
                -0.007 * np.sin(theta),
                0.06 + 0.03 * np.sin(theta * zrate)
            ]]).T
        elif scene == PhototourismScenes.BRANDENBURG:
            translation = c2w[:, 3:4] + torch.tensor([[
                0.08 * np.cos(theta),
                -0.07 - 0.01 * np.sin(theta),
                -0.0 + 0.1 * np.sin(theta * zrate)
            ]]).T
        elif scene == PhototourismScenes.TREVI:
            translation = c2w[:, 3:4] + torch.tensor([[
                -0.05 + 0.2 * np.cos(theta),
                -0.07 - 0.07 * np.sin(theta),
                0.02 + 0.05 * np.sin(theta * zrate)
            ]]).T
        else:
            raise NotImplementedError(scene)
        pose = torch.cat([rotation, translation], dim=1)
        render_poses.append(pose)
    return torch.stack(render_poses, dim=0)


def pt_render_poses(datadir: str, n_frames: int, frame_h: int = 800, frame_w: int = 800):
    scene = PhototourismScenes.get_scene_from_datadir(datadir)
    idx, _ = get_ids_for_split(datadir, split='train')
    train_poses, kinvs, bounds, res = load_camera_metadata(datadir, idx)
    train_poses, kinvs, bounds = scale_cam_metadata(train_poses, kinvs, bounds, scale=0.05)

    # build camera intrinsic from appropriate focal distance and cx, cy. Good for trevi
    k = np.array([[780.0, 0, -frame_w / 2], [0, -780, -frame_h / 2], [0, 0, -1]])
    kinv = torch.from_numpy(np.linalg.inv(k)).to(torch.float32)

    bounds = torch.from_numpy(bounds).float()
    train_poses = torch.from_numpy(train_poses).float()

    r_poses = pt_spiral_path(scene, train_poses, n_frames=n_frames, zrate=1, n_rots=1)

    all_rays_o, all_rays_d, camera_ids, near_fars = [], [], [], []
    for pose_id, pose in enumerate(r_poses):
        pose = pose.float()
        rays_o, rays_d = get_rays_tourism(frame_h, frame_w, kinv, pose)
        rays_o = rays_o.view(-1, 3)
        rays_d = rays_d.view(-1, 3)
        all_rays_o.append(rays_o)
        all_rays_d.append(rays_d)

        # Find the closest cam
        closest_cam_idx = torch.linalg.norm(
            train_poses[:, :3, :].view(train_poses.shape[0], -1) - pose.view(-1), dim=1
        ).argmin()

        # For brandenburg and trevi
        if scene == PhototourismScenes.BRANDENBURG or scene == PhototourismScenes.TREVI:
            near_fars.append((
                bounds[closest_cam_idx] + torch.tensor([0.01, 0.0])
            ))
        elif scene == PhototourismScenes.SACRE:
            near_fars.append((
                bounds[closest_cam_idx] + torch.tensor([0.05, 0.0])
            ))
    # camera-IDs. They are floats interpolating between 2 appearance embeddings.
    x = torch.linspace(-1, 1, len(r_poses))
    s = 0.3
    camera_ids = 1/(s * math.sqrt(2 * torch.pi)) * torch.exp(- (x)**2 / (2 * s**2))
    camera_ids = (camera_ids - camera_ids.min()) / camera_ids.max()
    all_rays_o = torch.stack(all_rays_o, dim=0)
    all_rays_d = torch.stack(all_rays_d, dim=0)
    near_fars = torch.stack(near_fars, dim=0)
    return all_rays_o, all_rays_d, camera_ids, near_fars
