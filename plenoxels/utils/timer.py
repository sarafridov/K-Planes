"""Accurate timer for CUDA code"""
import torch.cuda


class CudaTimer:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.start, self.end = None, None
        self.timings = {}
        if self.enabled:
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
            self.prev_time_gpu = self.start.record()

    def reset(self):
        if self.enabled:
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
            self.prev_time_gpu = self.start.record()

    def check(self, name=None):
        if self.enabled:
            self.end.record()
            torch.cuda.synchronize()
            gpu_time = self.start.elapsed_time(self.end)
            self.timings[name] = gpu_time

            self.prev_time_gpu = self.start.record()
