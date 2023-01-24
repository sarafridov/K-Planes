"""Entry point for simple renderings, given a trainer and some poses."""
import os
import logging as log
from typing import Union

import torch

from plenoxels.models.lowrank_model import LowrankModel
from plenoxels.utils.my_tqdm import tqdm
from plenoxels.ops.image.io import write_video_to_file
from plenoxels.runners.static_trainer import StaticTrainer
from plenoxels.runners.video_trainer import VideoTrainer


@torch.no_grad()
def render_to_path(trainer: Union[VideoTrainer, StaticTrainer], extra_name: str = "") -> None:
    """Render all poses in the `test_dataset`, saving them to file
    Args:
        trainer: The trainer object which is used for rendering
        extra_name: String to append to the saved file-name
    """
    dataset = trainer.test_dataset

    pb = tqdm(total=100, desc=f"Rendering scene")
    frames = []
    for img_idx, data in enumerate(dataset):
        ts_render = trainer.eval_step(data)

        if isinstance(dataset.img_h, int):
            img_h, img_w = dataset.img_h, dataset.img_w
        else:
            img_h, img_w = dataset.img_h[img_idx], dataset.img_w[img_idx]
        preds_rgb = (
            ts_render["rgb"]
            .reshape(img_h, img_w, 3)
            .cpu()
            .clamp(0, 1)
            .mul(255.0)
            .byte()
            .numpy()
        )
        frames.append(preds_rgb)
        pb.update(1)
    pb.close()

    out_fname = os.path.join(trainer.log_dir, f"rendering_path_{extra_name}.mp4")
    write_video_to_file(out_fname, frames)
    log.info(f"Saved rendering path with {len(frames)} frames to {out_fname}")


def normalize_for_disp(img):
    img = img - torch.min(img)
    img = img / torch.max(img)
    return img


@torch.no_grad()
def decompose_space_time(trainer: StaticTrainer, extra_name: str = "") -> None:
    """Render space-time decomposition videos for poses in the `test_dataset`.

    The space-only part of the decomposition is obtained by setting the time-planes to 1.
    The time-only part is obtained by simple subtraction of the space-only part from the full
    rendering.

    Args:
        trainer: The trainer object which is used for rendering
        extra_name: String to append to the saved file-name
    """
    chosen_cam_idx = 15
    model: LowrankModel = trainer.model
    dataset = trainer.test_dataset

    # Store original parameters from main field and proposal-network field
    parameters = []
    for multires_grids in model.field.grids:
        parameters.append([grid.data for grid in multires_grids])
    pn_parameters = []
    for pn in model.proposal_networks:
        pn_parameters.append([grid_plane.data for grid_plane in pn.grids])

    camdata = None
    for img_idx, data in enumerate(dataset):
        if img_idx == chosen_cam_idx:
            camdata = data
    if camdata is None:
        raise ValueError(f"Cam idx {chosen_cam_idx} invalid.")

    num_frames = img_idx + 1
    frames = []
    for img_idx in tqdm(range(num_frames), desc="Rendering scene with separate space and time components"):
        # Linearly interpolated timestamp, normalized between -1, 1
        camdata["timestamps"] = torch.Tensor([img_idx / num_frames]) * 2 - 1

        if isinstance(dataset.img_h, int):
            img_h, img_w = dataset.img_h, dataset.img_w
        else:
            img_h, img_w = dataset.img_h[img_idx], dataset.img_w[img_idx]

        # Full model: turn on time-planes
        for i in range(len(model.field.grids)):
            for plane_idx in [2, 4, 5]:
                model.field.grids[i][plane_idx].data = parameters[i][plane_idx]
        for i in range(len(model.proposal_networks)):
            for plane_idx in [2, 4, 5]:
                model.proposal_networks[i].grids[plane_idx].data = pn_parameters[i][plane_idx]
        preds = trainer.eval_step(camdata)
        full_out = preds["rgb"].reshape(img_h, img_w, 3).cpu()

        # Space-only model: turn off time-planes
        for i in range(len(model.field.grids)):
            for plane_idx in [2, 4, 5]:  # time-grids off
                model.field.grids[i][plane_idx].data = torch.ones_like(parameters[i][plane_idx])
        for i in range(len(model.proposal_networks)):
            for plane_idx in [2, 4, 5]:
                model.proposal_networks[i].grids[plane_idx].data = torch.ones_like(pn_parameters[i][plane_idx])
        preds = trainer.eval_step(camdata)
        spatial_out = preds["rgb"].reshape(img_h, img_w, 3).cpu()

        # Temporal model: full - space
        temporal_out = normalize_for_disp(full_out - spatial_out)

        frames.append(
            torch.cat([full_out, spatial_out, temporal_out], dim=1)
                 .clamp(0, 1)
                 .mul(255.0)
                 .byte()
                 .numpy()
        )

    out_fname = os.path.join(trainer.log_dir, f"spacetime_{extra_name}.mp4")
    write_video_to_file(out_fname, frames)
    log.info(f"Saved rendering path with {len(frames)} frames to {out_fname}")
