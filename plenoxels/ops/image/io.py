import glob
import os
from typing import List, Optional

import cv2
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import logging as log
import numpy as np
import imageio.v3 as iio


def write_png(path, data):
    """Writes an PNG image to some path.

    Args:
        path (str): Path to save the PNG.
        data (np.array): HWC image.

    Returns:
        (void): Writes to path.
    """
    Image.fromarray(data).save(path)


def read_png(file_name: str, resize_h: Optional[int] = None, resize_w: Optional[int] = None) -> torch.Tensor:
    """Reads a PNG image from path, potentially resizing it.
    """
    img = Image.open(file_name).convert('RGB')  # PIL outputs BGR by default
    if resize_h is not None and resize_w is not None:
        img.resize((resize_w, resize_h), Image.LANCZOS)
    img = TF.to_tensor(img)  # TF converts to C, H, W
    img = img.permute(1, 2, 0).contiguous()  # H, W, C
    return img


def glob_imgs(path, exts=None):
    """Utility to find images in some path.

    Args:
        path (str): Path to search images in.
        exts (list of str): List of extensions to try.

    Returns:
        (list of str): List of paths that were found.
    """
    if exts is None:
        exts = ['*.png', '*.PNG', '*.jpg', '*.jpeg', '*.JPG', '*.JPEG']
    imgs = []
    for ext in exts:
        imgs.extend(glob.glob(os.path.join(path, ext)))
    return imgs


def write_video_to_file(file_name, frames: List[np.ndarray]):
    log.info(f"Saving video ({len(frames)} frames) to {file_name}")
    # Photo tourism image sizes differ
    sizes = np.array([frame.shape[:2] for frame in frames])
    same_size_frames = np.unique(sizes, axis=0).shape[0] == 1
    if same_size_frames:
        height, width = frames[0].shape[:2]
        video = cv2.VideoWriter(
            file_name, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
        for img in frames:
            if isinstance(img, torch.Tensor):
                img = img.numpy()
            video.write(img[:, :, ::-1])  # opencv uses BGR instead of RGB
        cv2.destroyAllWindows()
        video.release()
    else:
        height = sizes[:, 0].max()
        width = sizes[:, 1].max()
        video = cv2.VideoWriter(
            file_name, cv2.VideoWriter_fourcc(*'mp4v'), 5, (width, height))
        for img in frames:
            image = np.zeros((height, width, 3), dtype=np.uint8)
            h, w = img.shape[:2]
            if isinstance(img, torch.Tensor):
                img = img.numpy()
            image[(height-h)//2:(height-h)//2+h, (width-w)//2:(width-w)//2+w, :] = img
            video.write(image[:, :, ::-1])  # opencv uses BGR instead of RGB
        cv2.destroyAllWindows()
        video.release()


def read_mp4(file_name: str) -> List[torch.Tensor]:
    all_frames = iio.imread(
        file_name, plugin='pyav', format='rgb24', constant_framerate=True, thread_count=2
    )
    out_frames = [torch.from_numpy(f) for f in all_frames]
    return out_frames
