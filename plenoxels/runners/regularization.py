import abc
from typing import Sequence

import torch
import torch.optim.lr_scheduler
from torch import nn

from plenoxels.models.lowrank_model import LowrankModel
from plenoxels.ops.losses.histogram_loss import interlevel_loss
from plenoxels.raymarching.ray_samplers import RaySamples


def grid_from_model(model: LowrankModel, what: str) -> Sequence[nn.ParameterList]:
    if what == 'field':
        return model.field.grids
    elif what == 'proposal_network':
        return [p.grids for p in model.proposal_networks]
    else:
        raise NotImplementedError(what)


def compute_plane_tv(t: torch.Tensor, only_last_dim: bool = False) -> float:
    """Computes total variance across a plane.
    Args:
        t: Plane tensor
        only_last_dim: Whether to only compute total variance across the last dimension
    Returns:
        Total variance
    """
    _, h, w = t.shape
    w_tv = torch.square(t[..., :, 1:] - t[..., :, : w - 1]).mean()

    if only_last_dim:
        return w_tv

    h_tv = torch.square(t[..., 1:, :] - t[..., : h - 1, :]).mean()
    return h_tv + w_tv


def compute_plane_smoothness(t: torch.Tensor) -> float:
    """Computes smoothness across a time plane.
    Args:
        t: Plane tensor
    Returns:
        Time smoothness
    """
    h = t.shape[-2]
    # Convolve with a second derivative filter, in the time dimension which is dimension 2
    first_difference = t[..., 1:, :] - t[..., : h - 1, :]  # [c, h-1, w]
    second_difference = first_difference[..., 1:, :] - first_difference[..., : h - 2, :]  # [c, h-2, w]
    # Take the L2 norm of the result
    return torch.square(second_difference).mean()


class Regularizer:
    def __init__(self, reg_type, initialization):
        self.reg_type = reg_type
        self.initialization = initialization
        self.weight = float(self.initialization)
        self.last_reg = None

    def step(self, global_step):
        pass

    def report(self, d):
        if self.last_reg is not None:
            d[self.reg_type].update(self.last_reg.item())

    def regularize(self, *args, **kwargs) -> torch.Tensor:
        out = self._regularize(*args, **kwargs) * self.weight
        self.last_reg = out.detach()
        return out

    @abc.abstractmethod
    def _regularize(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError()

    def __str__(self):
        return f"Regularizer({self.reg_type}, weight={self.weight})"


class PlaneTV(Regularizer):
    """Computes total variance across each spatial plane in the grids.
    Args:
        initial_weight: default multiplier to control the amount of regularization
        what: Whether to the regularizer is applied to fields or proposal networks
    """
    def __init__(self, initial_weight, what: str = 'field') -> None:
        if what not in {'field', 'proposal_network'}:
            raise ValueError(f'what must be one of "field" or "proposal_network" '
                             f'but {what} was passed.')
        name = f'planeTV-{what[:2]}'
        super().__init__(name, initial_weight)
        self.what = what

    def step(self, global_step):
        pass

    def _regularize(self, model: LowrankModel, **kwargs) -> float:
        multi_res_grids = grid_from_model(model, self.what)
        total = 0.0
        num_planes = 0
        for grids in multi_res_grids:
            spatial_planes = {0, 1, 2} if len(grids) == 3 else {0, 1, 3}
            for grid_id, grid in enumerate(grids):
                if grid_id in spatial_planes:
                    total += compute_plane_tv(grid)
                else:
                    # Space is the last dimension for space-time planes.
                    total += compute_plane_tv(grid, only_last_dim=True)
                num_planes += 1
        return total / num_planes


class TimeSmoothness(Regularizer):
    """Computes smoothness across each time plane in the grids.
    Args:
        initial_weight: default multiplier to control the amount of regularization
        what: Whether to the regularizer is applied to fields or proposal networks
    """
    def __init__(self, initial_weight, what: str = 'field') -> None:
        if what not in {'field', 'proposal_network'}:
            raise ValueError(f'what must be one of "field" or "proposal_network" '
                             f'but {what} was passed.')
        name = f'time-smooth-{what[:2]}'
        super().__init__(name, initial_weight)
        self.what = what

    def _regularize(self, model: LowrankModel, **kwargs) -> float:
        multi_res_grids = grid_from_model(model, self.what)
        total = 0.0
        num_planes = 0
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        for grids in multi_res_grids:
            time_planes = [] if len(grids) == 3 else [2, 4, 5]
            for plane_id in time_planes:
                total += compute_plane_smoothness(grids[plane_id])
                num_planes += 1
        return total / num_planes


class L1TimePlanes(Regularizer):
    """Computes the L1 distance from the multiplicative identity (1) for spatiotemporal planes.
    Args:
        initial_weight: default multiplier to control the amount of regularization
        what: Whether to the regularizer is applied to fields or proposal networks
    """
    def __init__(self, initial_weight, what='field'):
        if what not in {'field', 'proposal_network'}:
            raise ValueError(f'what must be one of "field" or "proposal_network" '
                             f'but {what} was passed.')
        super().__init__(f'l1-time-{what[:2]}', initial_weight)
        self.what = what

    def _regularize(self, model: LowrankModel, **kwargs) -> float:
        spatiotemporal_planes = [2, 4, 5]
        multi_res_grids = grid_from_model(model, self.what)
        total = 0.0
        num_planes = 0
        for grids in multi_res_grids:
            if len(grids) == 3:
                continue
            for grid_id in spatiotemporal_planes:
                total += torch.abs(1 - grids[grid_id]).mean()
                num_planes += 1
        return total / num_planes


class HistogramLoss(Regularizer):
    def __init__(self, initial_weight):
        super().__init__('histogram-loss', initial_weight)

    def _regularize(self, model: LowrankModel, model_out, **kwargs) -> torch.Tensor:
        return interlevel_loss(model_out['weights_list'], model_out['ray_samples_list'])


class DistortionLoss(Regularizer):
    def __init__(self, initial_weight):
        super().__init__('distortion-loss', initial_weight)

    def _regularize(self, model: LowrankModel, model_out, **kwargs) -> torch.Tensor:
        """
        Efficient O(N) realization of distortion loss.
        from https://github.com/sunset1995/torch_efficient_distloss/blob/main/torch_efficient_distloss/eff_distloss.py
        There are B rays each with N sampled points.
        """
        w = model_out['weights_list'][-1]
        rs: RaySamples = model_out['ray_samples_list'][-1]
        m = (rs.starts + rs.ends) / 2
        interval = rs.deltas

        loss_uni = (1/3) * (interval * w.pow(2)).sum(dim=-1).mean()
        wm = (w * m)
        w_cumsum = w.cumsum(dim=-1)
        wm_cumsum = wm.cumsum(dim=-1)
        loss_bi_0 = wm[..., 1:] * w_cumsum[..., :-1]
        loss_bi_1 = w[..., 1:] * wm_cumsum[..., :-1]
        loss_bi = 2 * (loss_bi_0 - loss_bi_1).sum(dim=-1).mean()
        return loss_bi + loss_uni
