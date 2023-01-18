import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

__all__ = (
    "trunc_exp",
    "init_density_activation",
)

from torch.nn import functional as F


class TruncatedExponential(Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, min=-15, max=15))


trunc_exp = TruncatedExponential.apply


def init_density_activation(activation_type: str):
    if activation_type == 'trunc_exp':
        return lambda x: trunc_exp(x - 1)
    elif activation_type == 'relu':
        return F.relu
    else:
        raise ValueError(activation_type)
