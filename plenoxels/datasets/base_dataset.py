from abc import ABC
import os
from typing import Optional, List, Union

import torch
from torch.utils.data import Dataset

from .intrinsics import Intrinsics


class BaseDataset(Dataset, ABC):
    def __init__(self,
                 datadir: str,
                 scene_bbox: torch.Tensor,
                 split: str,
                 is_ndc: bool,
                 is_contracted: bool,
                 rays_o: Optional[torch.Tensor],
                 rays_d: Optional[torch.Tensor],
                 intrinsics: Union[Intrinsics, List[Intrinsics]],
                 batch_size: Optional[int] = None,
                 imgs: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
                 sampling_weights: Optional[torch.Tensor] = None,
                 weights_subsampled: int = 1,
                 ):
        self.datadir = datadir
        self.name = os.path.basename(self.datadir)
        self.scene_bbox = scene_bbox
        self.split = split
        self.is_ndc = is_ndc
        self.is_contracted = is_contracted
        self.weights_subsampled = weights_subsampled
        self.batch_size = batch_size
        if self.split == 'train':
            assert self.batch_size is not None
        self.rays_o = rays_o
        self.rays_d = rays_d
        self.imgs = imgs
        if self.imgs is not None:
            self.num_samples = len(self.imgs)
        elif self.rays_o is not None:
            self.num_samples = len(self.rays_o)
        else:
            self.num_samples = None
            #raise RuntimeError("Can't figure out num_samples.")
        self.intrinsics = intrinsics
        self.sampling_weights = sampling_weights
        if self.sampling_weights is not None:
            assert len(self.sampling_weights) == self.num_samples, (
                f"Expected {self.num_samples} sampling weights but given {len(self.sampling_weights)}."
            )
        self.sampling_batch_size = 2_000_000  # Increase this?
        if self.num_samples is not None:
            self.use_permutation = self.num_samples < 100_000_000  # 64M is static
        else:
            self.use_permutation = True
        self.perm = None

    @property
    def img_h(self) -> Union[int, List[int]]:
        if isinstance(self.intrinsics, list):
            return [i.height for i in self.intrinsics]
        return self.intrinsics.height

    @property
    def img_w(self) -> Union[int, List[int]]:
        if isinstance(self.intrinsics, list):
            return [i.width for i in self.intrinsics]
        return self.intrinsics.width

    def reset_iter(self):
        if self.sampling_weights is None and self.use_permutation:
            self.perm = torch.randperm(self.num_samples)
        else:
            del self.perm
            self.perm = None

    def get_rand_ids(self, index):
        assert self.batch_size is not None, "Can't get rand_ids for test split"
        if self.sampling_weights is not None:
            batch_size = self.batch_size // (self.weights_subsampled ** 2)
            num_weights = len(self.sampling_weights)
            if num_weights > self.sampling_batch_size:
                # Take a uniform random sample first, then according to the weights
                subset = torch.randint(
                    0, num_weights, size=(self.sampling_batch_size,),
                    dtype=torch.int64, device=self.sampling_weights.device)
                samples = torch.multinomial(
                    input=self.sampling_weights[subset], num_samples=batch_size)
                return subset[samples]
            return torch.multinomial(
                input=self.sampling_weights, num_samples=batch_size)
        else:
            batch_size = self.batch_size
            if self.use_permutation:
                return self.perm[index * batch_size: (index + 1) * batch_size]
            else:
                return torch.randint(0, self.num_samples, size=(batch_size, ))

    def __len__(self):
        if self.split == 'train':
            return (self.num_samples + self.batch_size - 1) // self.batch_size
        else:
            return self.num_samples

    def __getitem__(self, index, return_idxs: bool = False):
        if self.split == 'train':
            index = self.get_rand_ids(index)
        out = {}
        if self.rays_o is not None:
            out["rays_o"] = self.rays_o[index]
        if self.rays_d is not None:
            out["rays_d"] = self.rays_d[index]
        if self.imgs is not None:
            out["imgs"] = self.imgs[index]
        else:
            out["imgs"] = None
        if return_idxs:
            return out, index
        return out
