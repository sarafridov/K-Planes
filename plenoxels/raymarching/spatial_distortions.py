from typing import Optional, Union

import torch
import torch.nn as nn


class SpatialDistortion(nn.Module):
    """Apply spatial distortions"""

    def forward(
        self, positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            positions: Sample to distort (shape: batch-size, ..., 3)
        Returns:
            distorted sample - same shape
        """


class SceneContraction(SpatialDistortion):
    """Contract unbounded space using the contraction was proposed in MipNeRF-360.
        We use the following contraction equation:
        .. math::
            f(x) = \\begin{cases}
                x & ||x|| \\leq 1 \\\\
                (2 - \\frac{1}{||x||})(\\frac{x}{||x||}) & ||x|| > 1
            \\end{cases}
        If the order is not specified, we use the Frobenius norm, this will contract the space to a sphere of
        radius 1. If the order is L_inf (order=float("inf")), we will contract the space to a cube of side length 2.
        If using voxel based encodings such as the Hash encoder, we recommend using the L_inf norm.
        Args:
            order: Order of the norm. Default to the Frobenius norm. Must be set to None for Gaussians.
    """

    def __init__(self,
                 order: Optional[Union[float, int]] = None,
                 global_translation: Optional[torch.Tensor] = None,
                 global_scale: Optional[torch.Tensor] = None,
                 ) -> None:
        super().__init__()
        self.order = order
        if global_translation is None:
            global_translation = torch.tensor([0.0, 0.0, 0.0])
        self.global_translation = nn.Parameter(global_translation, requires_grad=False)
        if global_scale is None:
            global_scale = torch.tensor([1.0, 1.0, 1.0])
            
        self.global_scale = nn.Parameter(global_scale, requires_grad=False)

    def forward(self, positions):
        # Apply global scale and translation
        positions = (
            positions * self.global_scale[None, None, :]
            + self.global_translation[None, None, :]
        )

        mag = torch.linalg.norm(positions, ord=self.order, dim=-1)
        mask = mag >= 1
        x_new = positions.clone()
        x_new[mask] = (2 - (1 / mag[mask][..., None])) * (positions[mask] / mag[mask][..., None])

        return x_new

    def __str__(self):
        return (f"SceneContraction(global_translation={self.global_translation}, "
                f"global_scale={self.global_scale})")
