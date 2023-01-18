"""The ray samplers are almost completely copied from NeRF-studio

https://github.com/nerfstudio-project/nerfstudio/blob/628e4fe1a638e7fb3b7ad33d4d91a4b1d63a9b68/nerfstudio/model_components/ray_samplers.py

Copyright 2022 The Nerfstudio Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional, Callable, Tuple, List

import torch
from torch import nn


@dataclass
class RaySamples:
    """xyz coordinate for ray origin."""
    origins: torch.Tensor  # [bs:..., 3]
    """Direction of ray."""
    directions: torch.Tensor  # [bs:..., 3]
    """Where the frustum starts along a ray."""
    starts: torch.Tensor  # [bs:..., 1]
    """Where the frustum ends along a ray."""
    ends: torch.Tensor  # [bs:..., 1]
    """"width" of each sample."""
    deltas: Optional[torch.Tensor] = None  # [bs, ...?, 1]
    """Start of normalized bin edges along ray [0,1], before warping is applied, ie. linear in disparity sampling."""
    spacing_starts: Optional[torch.Tensor] = None  # [bs, ...?, num_samples, 1]
    """End of normalized bin edges along ray [0,1], before warping is applied, ie. linear in disparity sampling."""
    spacing_ends: Optional[torch.Tensor] = None  # [bs, ...?, num_samples, 1]
    """Function to convert bins to euclidean distance."""
    spacing_to_euclidean_fn: Optional[Callable] = None

    def get_positions(self) -> torch.Tensor:
        """Calulates "center" position of frustum. Not weighted by mass.
        Returns:
            xyz positions (..., 3).
        """
        return self.origins + self.directions * (self.starts + self.ends) / 2  # world space

    def get_weights2(self, densities: torch.Tensor) -> torch.Tensor:
        densities = densities.squeeze(2)
        deltas = self.deltas.squeeze(2)
        delta_mask = deltas > 0
        deltas = deltas[delta_mask]

        delta_density = torch.zeros_like(densities)
        delta_density[delta_mask] = deltas * densities[delta_mask]
        alphas = 1 - torch.exp(-delta_density)

        transmittance = torch.cat(
            (
                torch.ones(alphas.shape[0], 1, device=alphas.device),
                torch.cumprod(1.0 - alphas, dim=-1)
            ), dim=-1
        )
        weights = alphas * transmittance[:, :-1]
        return weights[..., None]

    def get_weights(self, densities: torch.Tensor) -> torch.Tensor:
        """Return weights based on predicted densities
        Args:
            densities: Predicted densities for samples along ray (..., num_samples, 1)
        Returns:
            Weights for each sample  (..., num_samples, 1)
        """
        delta_mask = self.deltas > 0
        deltas = self.deltas[delta_mask]

        delta_density = torch.zeros_like(densities)
        delta_density[delta_mask] = deltas * densities[delta_mask]
        alphas = 1 - torch.exp(-delta_density)

        transmittance = torch.cumsum(delta_density[..., :-1, :], dim=-2)
        transmittance = torch.cat(
            [torch.zeros((*transmittance.shape[:1], 1, 1), device=densities.device), transmittance], dim=-2
        )
        transmittance = torch.exp(-transmittance)  # [..., "num_samples"]
        weights = alphas * transmittance  # [..., "num_samples"]
        return weights


@dataclass
class RayBundle:
    """A bundle of ray parameters."""

    """Ray origins (XYZ)"""
    origins: torch.Tensor  # [..., 3]
    """Unit ray direction vector"""
    directions: torch.Tensor  # [..., 3]
    """Distance along ray to start sampling"""
    nears: Optional[torch.Tensor] = None  # [..., 1]
    """Rays Distance along ray to stop sampling"""
    fars: Optional[torch.Tensor] = None  # [..., 1]

    def __len__(self):
        num_rays = torch.numel(self.origins) // self.origins.shape[-1]
        return num_rays

    def get_ray_samples(
        self,
        bin_starts: torch.Tensor,
        bin_ends: torch.Tensor,
        spacing_starts: Optional[torch.Tensor] = None,
        spacing_ends: Optional[torch.Tensor] = None,
        spacing_to_euclidean_fn: Optional[Callable] = None,
    ) -> RaySamples:
        """Produces samples for each ray by projection points along the ray direction. Currently samples uniformly.
        Args:
            bin_starts: Distance from origin to start of bin.
                TensorType["bs":..., "num_samples", 1]
            bin_ends: Distance from origin to end of bin.
        Returns:
            Samples projected along ray.
        """
        deltas = bin_ends - bin_starts
        return RaySamples(
            origins=self.origins[..., None, :],  # [..., 1, 3]
            directions=self.directions[..., None, :],  # [..., 1, 3]
            starts=bin_starts,  # [..., num_samples, 1]  world
            ends=bin_ends,  # [..., num_samples, 1]      world
            deltas=deltas,  # [..., num_samples, 1]  world coo
            spacing_starts=spacing_starts,  # [..., num_samples, 1]
            spacing_ends=spacing_ends,  # [..., num_samples, 1]
            spacing_to_euclidean_fn=spacing_to_euclidean_fn,
        )


class Sampler(nn.Module):
    """Generate Samples
    Args:
        num_samples: number of samples to take
    """

    def __init__(
        self,
        num_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples

    @abstractmethod
    def generate_ray_samples(self) -> RaySamples:
        """Generate Ray Samples"""

    def forward(self, *args, **kwargs) -> RaySamples:
        """Generate ray samples"""
        return self.generate_ray_samples(*args, **kwargs)


class SpacedSampler(Sampler):
    """Sample points according to a function.
    Args:
        num_samples: Number of samples per ray
        spacing_fn: Function that dictates sample spacing (ie `lambda x : x` is uniform).
        spacing_fn_inv: The inverse of spacing_fn.
        train_stratified: Use stratified sampling during training. Defults to True
        single_jitter: Use a same random jitter for all samples along a ray. Defaults to False
    """

    def __init__(
        self,
        spacing_fn: Callable,
        spacing_fn_inv: Callable,
        num_samples: Optional[int] = None,
        train_stratified=True,
        single_jitter=False,
    ) -> None:
        super().__init__(num_samples=num_samples)
        self.train_stratified = train_stratified
        self.single_jitter = single_jitter
        self.spacing_fn = spacing_fn
        self.spacing_fn_inv = spacing_fn_inv

    # noinspection PyMethodOverriding
    def generate_ray_samples(
        self,
        ray_bundle: RayBundle,
        num_samples: Optional[int] = None,
    ) -> RaySamples:
        """Generates position samples accoring to spacing function.
        Args:
            ray_bundle: Ray-origins, directions, etc.
            num_samples: Number of samples per ray
        Returns:
            Positions and deltas for samples along a ray
        """
        num_samples = num_samples or self.num_samples
        assert num_samples is not None
        num_rays = ray_bundle.origins.shape[0]

        bins = torch.linspace(0.0, 1.0, num_samples + 1).to(ray_bundle.origins.device)[None, ...]  # [1, num_samples+1]

        # TODO More complicated than it needs to be.
        if self.train_stratified and self.training:
            if self.single_jitter:
                t_rand = torch.rand((num_rays, 1), dtype=bins.dtype, device=bins.device)
            else:
                t_rand = torch.rand((num_rays, num_samples + 1), dtype=bins.dtype, device=bins.device)
            bin_centers = (bins[..., 1:] + bins[..., :-1]) / 2.0
            bin_upper = torch.cat([bin_centers, bins[..., -1:]], -1)
            bin_lower = torch.cat([bins[..., :1], bin_centers], -1)
            bins = bin_lower + (bin_upper - bin_lower) * t_rand
        else:
            bins = bins.repeat(num_rays, 1)

        # s_near, s_far in [0, 1]
        s_near, s_far = (self.spacing_fn(x) for x in (ray_bundle.nears, ray_bundle.fars))
        spacing_to_euclidean_fn = lambda x: self.spacing_fn_inv(x * s_far + (1 - x) * s_near)
        # euclidean = world
        euclidean_bins = spacing_to_euclidean_fn(bins)  # [num_rays, num_samples+1]

        return ray_bundle.get_ray_samples(
            bin_starts=euclidean_bins[..., :-1, None],  # world [near, far]
            bin_ends=euclidean_bins[..., 1:, None],     # world [near, far]
            spacing_starts=bins[..., :-1, None],        # [0, 1]
            spacing_ends=bins[..., 1:, None],           # [0, 1]
            spacing_to_euclidean_fn=spacing_to_euclidean_fn,
        )


class UniformSampler(SpacedSampler):
    """Sample uniformly along a ray
    Args:
        num_samples: Number of samples per ray
        train_stratified: Use stratified sampling during training. Defults to True
        single_jitter: Use a same random jitter for all samples along a ray. Defaults to False
    """

    def __init__(
        self,
        num_samples: Optional[int] = None,
        train_stratified=True,
        single_jitter=False,
    ) -> None:
        super().__init__(
            num_samples=num_samples,
            spacing_fn=lambda x: x,
            spacing_fn_inv=lambda x: x,
            train_stratified=train_stratified,
            single_jitter=single_jitter,
        )


class LinearDisparitySampler(SpacedSampler):
    """Sample linearly in disparity along a ray
    Args:
        num_samples: Number of samples per ray
        train_stratified: Use stratified sampling during training. Defults to True
        single_jitter: Use a same random jitter for all samples along a ray. Defaults to False
    """

    def __init__(
        self,
        num_samples: Optional[int] = None,
        train_stratified=True,
        single_jitter=False,
    ) -> None:
        super().__init__(
            num_samples=num_samples,
            spacing_fn=lambda x: 1 / x,
            spacing_fn_inv=lambda x: 1 / x,
            train_stratified=train_stratified,
            single_jitter=single_jitter,
        )


class UniformLinDispPiecewiseSampler(SpacedSampler):
    """Piecewise sampler along a ray that allocates the first half of the samples uniformly and the second half
    using linearly in disparity spacing.
    Args:
        num_samples: Number of samples per ray
        train_stratified: Use stratified sampling during training. Defults to True
        single_jitter: Use a same random jitter for all samples along a ray. Defaults to False
    """

    def __init__(
        self,
        num_samples: Optional[int] = None,
        train_stratified=True,
        single_jitter=False,
    ) -> None:
        super().__init__(
            num_samples=num_samples,
            spacing_fn=lambda x: torch.where(x < 1, x / 2, 1 - 1 / (2 * x)),
            spacing_fn_inv=lambda x: torch.where(x < 0.5, 2 * x, 1 / (2 - 2 * x)),
            train_stratified=train_stratified,
            single_jitter=single_jitter,
        )


class PDFSampler(Sampler):
    """Sample based on probability distribution
    Args:
        num_samples: Number of samples per ray
        train_stratified: Randomize location within each bin during training.
        single_jitter: Use a same random jitter for all samples along a ray. Defaults to False
        include_original: Add original samples to ray.
        histogram_padding: Amount to weights prior to computing PDF.
    """

    def __init__(
        self,
        num_samples: Optional[int] = None,
        train_stratified: bool = True,
        single_jitter: bool = False,
        include_original: bool = True,
        histogram_padding: float = 0.01,
    ) -> None:
        super().__init__(num_samples=num_samples)
        self.train_stratified = train_stratified
        self.include_original = include_original
        self.histogram_padding = histogram_padding
        self.single_jitter = single_jitter

    # noinspection PyMethodOverriding
    def generate_ray_samples(
        self,
        ray_bundle: RayBundle,
        ray_samples: Optional[RaySamples] = None,
        weights: Optional[torch.Tensor] = None,
        num_samples: Optional[int] = None,
        eps: float = 1e-5,
    ) -> RaySamples:
        """Generates position samples given a distribution.
        Args:
            ray_bundle: Ray-origins, directions, etc.
            ray_samples: Existing ray samples
            weights: Weights for each bin  [..., "num_samples", 1]
            num_samples: Number of samples per ray
            eps: Small value to prevent numerical issues.
        Returns:
            Positions and deltas for samples along a ray
        """
        if ray_samples is None or ray_bundle is None:
            raise ValueError("ray_samples must be provided")

        num_samples = num_samples or self.num_samples
        assert num_samples is not None
        num_bins = num_samples + 1

        weights = weights[..., 0] + self.histogram_padding

        # Add small offset to rays with zero weight to prevent NaNs
        weights_sum = torch.sum(weights, dim=-1, keepdim=True)
        padding = torch.relu(eps - weights_sum)
        weights = weights + padding / weights.shape[-1]
        weights_sum += padding

        pdf = weights / weights_sum
        cdf = torch.min(torch.ones_like(pdf), torch.cumsum(pdf, dim=-1))
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

        if self.train_stratified and self.training:
            # Stratified samples between 0 and 1
            u = torch.linspace(0.0, 1.0 - (1.0 / num_bins), steps=num_bins, device=cdf.device)
            u = u.expand((*cdf.shape[:-1], num_bins))
            if self.single_jitter:
                rand = torch.rand((*cdf.shape[:-1], 1), device=cdf.device) / num_bins
            else:
                rand = torch.rand((*cdf.shape[:-1], num_samples + 1), device=cdf.device) / num_bins
            u = u + rand
        else:
            # Uniform samples between 0 and 1
            u = torch.linspace(0.0, 1.0 - (1.0 / num_bins), steps=num_bins, device=cdf.device)
            u = u + 1.0 / (2 * num_bins)
            u = u.expand(size=(*cdf.shape[:-1], num_bins))
        u = u.contiguous()

        assert (
            ray_samples.spacing_starts is not None and ray_samples.spacing_ends is not None
        ), "ray_sample spacing_starts and spacing_ends must be provided"
        assert ray_samples.spacing_to_euclidean_fn is not None, "ray_samples.spacing_to_euclidean_fn must be provided"
        existing_bins = torch.cat(
            [
                ray_samples.spacing_starts[..., 0],
                ray_samples.spacing_ends[..., -1:, 0],
            ],
            dim=-1,
        )  # [0, 1]

        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.clamp(inds - 1, 0, existing_bins.shape[-1] - 1)
        above = torch.clamp(inds, 0, existing_bins.shape[-1] - 1)
        cdf_g0 = torch.gather(cdf, -1, below)
        bins_g0 = torch.gather(existing_bins, -1, below)
        cdf_g1 = torch.gather(cdf, -1, above)
        bins_g1 = torch.gather(existing_bins, -1, above)

        t = torch.clip(torch.nan_to_num((u - cdf_g0) / (cdf_g1 - cdf_g0), 0), 0, 1)
        bins = bins_g0 + t * (bins_g1 - bins_g0)

        if self.include_original:
            bins, _ = torch.sort(torch.cat([existing_bins, bins], -1), -1)

        # Stop gradients
        bins = bins.detach()

        euclidean_bins = ray_samples.spacing_to_euclidean_fn(bins)

        return ray_bundle.get_ray_samples(
            bin_starts=euclidean_bins[..., :-1, None],
            bin_ends=euclidean_bins[..., 1:, None],
            spacing_starts=bins[..., :-1, None],
            spacing_ends=bins[..., 1:, None],
            spacing_to_euclidean_fn=ray_samples.spacing_to_euclidean_fn,
        )


class ProposalNetworkSampler(Sampler):
    """Sampler that uses a proposal network to generate samples."""
    def __init__(
        self,
        num_proposal_samples_per_ray: Tuple[int] = (64,),
        num_nerf_samples_per_ray: int = 32,
        num_proposal_network_iterations: int = 2,
        single_jitter: bool = False,
        update_sched: Callable = lambda x: 1,
        initial_sampler: Optional[Sampler] = None,
    ) -> None:
        super().__init__()
        self.num_proposal_samples_per_ray = num_proposal_samples_per_ray
        self.num_nerf_samples_per_ray = num_nerf_samples_per_ray
        self.num_proposal_network_iterations = num_proposal_network_iterations
        self.update_sched = update_sched
        if self.num_proposal_network_iterations < 1:
            raise ValueError("num_proposal_network_iterations must be >= 1")

        # samplers
        if initial_sampler is None:
            initial_sampler = UniformLinDispPiecewiseSampler(single_jitter=single_jitter)
        self.initial_sampler = initial_sampler
        self.pdf_sampler = PDFSampler(include_original=False, single_jitter=single_jitter)

        self._anneal = 1.0
        self._steps_since_update = 0
        self._step = 0

    def set_anneal(self, anneal: float) -> None:
        """Set the anneal value for the proposal network."""
        self._anneal = anneal

    def step_cb(self, step):
        """Callback to register a training step has passed. This is used to keep track of the sampling schedule"""
        self._step = step
        self._steps_since_update += 1

    def generate_ray_samples(
        self,
        ray_bundle: Optional[RayBundle] = None,
        timestamps: Optional[float] = None,
        density_fns: Optional[List[Callable]] = None,
    ) -> Tuple[RaySamples, List, List]:
        assert ray_bundle is not None
        assert density_fns is not None
        assert len(density_fns) == self.num_proposal_network_iterations

        weights_list = []
        ray_samples_list = []

        n = self.num_proposal_network_iterations
        weights = None
        ray_samples = None
        updated = self._steps_since_update > self.update_sched(self._step) or self._step < 10
        for i_level in range(n + 1):
            is_prop = i_level < n
            num_samples = self.num_proposal_samples_per_ray[i_level] if is_prop else self.num_nerf_samples_per_ray
            if i_level == 0:
                # Uniform sampling because we need to start with some samples
                ray_samples = self.initial_sampler(ray_bundle, num_samples=num_samples)
            else:
                # PDF sampling based on the last samples and their weights
                # Perform annealing to the weights. This will be a no-op if self._anneal is 1.0.
                assert weights is not None
                annealed_weights = torch.pow(weights, self._anneal)
                ray_samples = self.pdf_sampler(ray_bundle, ray_samples, annealed_weights, num_samples=num_samples)
            if is_prop:
                if updated:
                    # always update on the first step or the inf check in grad scaling crashes
                    density = density_fns[i_level](ray_samples.get_positions(), timestamps)  # world space
                else:
                    with torch.no_grad():
                        density = density_fns[i_level](ray_samples.get_positions(), timestamps)
                weights = ray_samples.get_weights(density)
                weights_list.append(weights)  # (num_rays, num_samples)
                ray_samples_list.append(ray_samples)
        if updated:
            self._steps_since_update = 0

        assert ray_samples is not None
        return ray_samples, weights_list, ray_samples_list

    def __str__(self):
        return (f"ProposalNetworkSampler("
                f"num_proposal_samples_per_ray={self.num_proposal_samples_per_ray}, "
                f"num_nerf_samples_per_ray={self.num_nerf_samples_per_ray}, "
                f"num_proposal_network_iterations={self.num_proposal_network_iterations})")
