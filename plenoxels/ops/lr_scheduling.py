import math

import torch
import torch.optim


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    eta_min: float = 0.0,
    num_cycles: float = 0.999,
    last_epoch: int = -1,
):
    """
    https://github.com/huggingface/transformers/blob/bd469c40659ce76c81f69c7726759d249b4aef49/src/transformers/optimization.py#L129
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(eta_min, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def get_log_linear_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    eta_min: float = 2e-5,
    eta_max: float = 1e-2,
    num_cycles: float = 0.999,
    last_epoch: int = -1,
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))  # in [0,1]
        return math.exp(progress * math.log(eta_min) + (1 - progress) * math.log(eta_max)) / eta_max

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def get_step_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    milestones,
    gamma: float,
    num_warmup_steps: int,
    last_epoch: int = -1,
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        out = 1.0
        for m in milestones:
            if current_step < m:
                break
            out *= gamma
        return out
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)
