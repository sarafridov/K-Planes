import logging as log
import math
import os
from collections import defaultdict
from typing import Dict, MutableMapping, Union, Any

import pandas as pd
import torch
import torch.utils.data

from plenoxels.datasets.phototourism_dataset import PhotoTourismDataset
from plenoxels.utils.ema import EMA
from plenoxels.models.lowrank_model import LowrankModel
from plenoxels.utils.my_tqdm import tqdm
from plenoxels.ops.image import metrics
from .base_trainer import BaseTrainer, init_dloader_random, initialize_model
from .regularization import (
    PlaneTV, TimeSmoothness, HistogramLoss, L1TimePlanes, DistortionLoss
)


class PhototourismTrainer(BaseTrainer):
    def __init__(self,
                 tr_loader: torch.utils.data.DataLoader,
                 tr_dset: torch.utils.data.TensorDataset,
                 ts_dset: torch.utils.data.TensorDataset,
                 num_steps: int,
                 logdir: str,
                 expname: str,
                 train_fp16: bool,
                 save_every: int,
                 valid_every: int,
                 save_outputs: bool,
                 device: Union[str, torch.device],
                 **kwargs
                 ):
        self.train_dataset = tr_dset
        self.test_dataset = ts_dset
        super().__init__(
            train_data_loader=tr_loader,
            num_steps=num_steps,
            logdir=logdir,
            expname=expname,
            train_fp16=train_fp16,
            save_every=save_every,
            valid_every=valid_every,
            save_outputs=save_outputs,
            device=device,
            **kwargs)

    def eval_step(self, data, **kwargs) -> MutableMapping[str, torch.Tensor]:
        """
        Note that here `data` contains a whole image. we need to split it up before tracing
        for memory constraints.
        """
        super().eval_step(data, **kwargs)
        batch_size = self.eval_batch_size
        with torch.cuda.amp.autocast(enabled=self.train_fp16), torch.no_grad():
            rays_o = data["rays_o"]
            rays_d = data["rays_d"]
            timestamp = data["timestamps"]
            near_far = data["near_fars"]
            bg_color = data["bg_color"]
            if isinstance(bg_color, torch.Tensor):
                bg_color = bg_color.to(self.device)
            preds = defaultdict(list)
            for b in range(math.ceil(rays_o.shape[0] / batch_size)):
                rays_o_b = rays_o[b * batch_size: (b + 1) * batch_size].to(self.device)
                rays_d_b = rays_d[b * batch_size: (b + 1) * batch_size].to(self.device)
                timestamps_b = timestamp[b * batch_size: (b + 1) * batch_size].to(self.device)
                near_far_b = near_far[b * batch_size: (b + 1) * batch_size].to(self.device)
                outputs = self.model(
                    rays_o_b, rays_d_b, timestamps=timestamps_b, bg_color=bg_color,
                    near_far=near_far_b)
                for k, v in outputs.items():
                    if "rgb" in k or "depth" in k:
                        preds[k].append(v.cpu())
        return {k: torch.cat(v, 0) for k, v in preds.items()}

    def train_step(self, data: Dict[str, Union[int, torch.Tensor]], **kwargs):
        return super().train_step(data, **kwargs)

    def post_step(self, progress_bar):
        return super().post_step(progress_bar)

    def pre_epoch(self):
        super().pre_epoch()
        # Reset randomness in train-dataset
        self.train_dataset.reset_iter()

    @torch.no_grad()
    def validate(self):
        with torch.autograd.enable_grad():
            self.optimize_appearance_codes()

        dataset = self.test_dataset
        per_scene_metrics = defaultdict(list)
        pred_frames, out_depths = [], []
        pb = tqdm(total=len(dataset), desc=f"Test scene ({dataset.name})")
        for img_idx, data in enumerate(dataset):
            preds = self.eval_step(data)
            out_metrics, out_img, out_depth = self.evaluate_metrics(
                data["imgs"], preds, dset=dataset, img_idx=img_idx, name=None,
                save_outputs=self.save_outputs)
            pred_frames.append(out_img)
            if out_depth is not None:
                out_depths.append(out_depth)
            for k, v in out_metrics.items():
                per_scene_metrics[k].append(v)
            pb.set_postfix_str(f"PSNR={out_metrics['psnr']:.2f}", refresh=False)
            pb.update(1)
        pb.close()
        val_metrics = [
            self.report_test_metrics(per_scene_metrics, extra_name=None),
        ]
        df = pd.DataFrame.from_records(val_metrics)
        df.to_csv(os.path.join(self.log_dir, f"test_metrics_step{self.global_step}.csv"))

    def calc_metrics(self, preds: torch.Tensor, gt: torch.Tensor):
        """
        Compute error metrics. This function gets called by `evaluate_metrics` in the base
        trainer class.
        :param preds:
        :param gt:
        :return:
        """
        mid = gt.shape[1] // 2
        gt_right = gt[:, mid:]
        preds_rgb_right = preds[:, mid:]

        err = (gt_right - preds_rgb_right) ** 2
        return {
            "mse": torch.mean(err),
            "psnr": metrics.psnr(preds_rgb_right, gt_right),
            "ssim": metrics.ssim(preds_rgb_right, gt_right),
            "ms-ssim": metrics.msssim(preds_rgb_right, gt_right),
        }

    def init_epoch_info(self):
        ema_weight = 0.9
        loss_info = defaultdict(lambda: EMA(ema_weight))
        return loss_info

    def init_model(self, **kwargs) -> LowrankModel:
        return initialize_model(self, **kwargs)

    def get_regularizers(self, **kwargs):
        return [
            PlaneTV(kwargs.get('plane_tv_weight', 0.0), what='field'),
            PlaneTV(kwargs.get('plane_tv_weight_proposal_net', 0.0), what='proposal_network'),
            L1TimePlanes(kwargs.get('l1_time_planes', 0.0), what='field'),
            L1TimePlanes(kwargs.get('l1_time_planes_proposal_net', 0.0), what='proposal_network'),
            TimeSmoothness(kwargs.get('time_smoothness_weight', 0.0), what='field'),
            TimeSmoothness(kwargs.get('time_smoothness_weight_proposal_net', 0.0), what='proposal_network'),
            HistogramLoss(kwargs.get('histogram_loss_weight', 0.0)),
            DistortionLoss(kwargs.get('distortion_loss_weight', 0.0)),
        ]

    @property
    def calc_metrics_every(self):
        return 5

    def optimize_appearance_step(self, data, im_id):
        rays_o = data["rays_o_left"]
        rays_d = data["rays_d_left"]
        imgs = data["imgs_left"]
        near_far = data["near_fars"]
        bg_color = data["bg_color"]
        if isinstance(bg_color, torch.Tensor):
            bg_color = bg_color.to(self.device)

        epochs = self.extra_args['app_optim_n_epochs']
        batch_size = self.eval_batch_size
        n_steps = math.ceil(rays_o.shape[0] / batch_size)

        camera_id = torch.full((batch_size, ), fill_value=im_id, dtype=torch.int32, device=self.device)

        app_optim = torch.optim.Adam(params=self.model.field.test_appearance_embedding.parameters(), lr=self.extra_args['app_optim_lr'])
        lr_sched = torch.optim.lr_scheduler.StepLR(app_optim, step_size=3 * n_steps, gamma=0.1)
        lowest_loss, lowest_loss_count = 100_000_000, 0
        grad_scaler = torch.cuda.amp.GradScaler(enabled=self.train_fp16)
        for n in range(epochs):
            idx = torch.randperm(rays_o.shape[0])
            for b in range(n_steps):
                batch_ids = idx[b * batch_size: (b + 1) * batch_size]
                rays_o_b = rays_o[batch_ids].to(self.device)
                rays_d_b = rays_d[batch_ids].to(self.device)
                imgs_b = imgs[batch_ids].to(self.device)
                near_far_b = near_far[batch_ids].to(self.device)
                camera_id_b = camera_id[:len(batch_ids)]

                with torch.cuda.amp.autocast(enabled=self.train_fp16):
                    fwd_out = self.model(
                        rays_o_b, rays_d_b, timestamps=camera_id_b, bg_color=bg_color,
                        near_far=near_far_b)
                    recon_loss = self.criterion(fwd_out['rgb'], imgs_b)
                app_optim.zero_grad(set_to_none=True)
                grad_scaler.scale(recon_loss).backward()
                grad_scaler.step(app_optim)
                grad_scaler.update()
                lr_sched.step()

                self.writer.add_scalar(
                    f"appearance_loss_{self.global_step}/recon_loss_{im_id}", recon_loss.item(),
                    b + n * n_steps)

                if recon_loss.item() < lowest_loss:
                    lowest_loss = recon_loss.item()
                    lowest_loss_count = 0
                lowest_loss_count += 1
            # 1 epoch without improvement -> stop
            if lowest_loss_count > 1 * n_steps:
                break

    def optimize_appearance_codes(self):
        """Optimize the appearance embedding of all test poses.

        At test time, since the correct embedding for a test pose is unknown, half of the test
        ground-truth image is used to optimize the appearance embedding corresponding to that test
        image. Evaluation is then performed on the other half of the image.
        """
        dset = self.test_dataset
        num_test_imgs = len(dset)

        # 1. Initialize test appearance code to average code.
        if not hasattr(self.model.field, "test_appearance_embedding"):
            tst_embedding = torch.nn.Embedding(
                num_test_imgs, self.model.field.appearance_embedding_dim
            ).to(self.device)
            with torch.autograd.no_grad():
                tst_embedding.weight.copy_(
                    self.model.field.appearance_embedding.weight
                        .detach()
                        .mean(dim=0, keepdim=True)
                        .expand(num_test_imgs, -1)
                )
            self.model.field.test_appearance_embedding = tst_embedding

        # 2. Setup parameter trainability
        self.model.eval()
        param_trainable = {}
        for pn, p in self.model.named_parameters():
            param_trainable[pn] = p.requires_grad
            p.requires_grad_(False)
        self.model.field.test_appearance_embedding.requires_grad_(True)

        # 3. Optimize
        pb = tqdm(total=len(dset), desc=f"Test-time appearance-code optimization")
        for img_idx, data in enumerate(dset):
            self.optimize_appearance_step(data, img_idx)
            pb.update(1)
        pb.close()

        # 4. Reset parameter trainability
        for pn, p in self.model.named_parameters():
            p.requires_grad_(param_trainable[pn])
        self.model.field.test_appearance_embedding.requires_grad_(False)


def init_tr_data(data_downsample, data_dir, **kwargs):
    batch_size = kwargs['batch_size']
    log.info(f"Loading PhotoTourismDataset with downsample={data_downsample}")
    tr_dset = PhotoTourismDataset(
        data_dir, split='train', batch_size=batch_size,
        contraction=kwargs['contract'], ndc=kwargs['ndc'],
        scene_bbox=kwargs['scene_bbox'], global_scale=kwargs.get('global_scale', None),
        global_translation=kwargs.get('global_translation', None),
    )
    tr_loader = torch.utils.data.DataLoader(
        tr_dset, batch_size=None, num_workers=4,  prefetch_factor=4, pin_memory=True,
        worker_init_fn=init_dloader_random)
    return {"tr_loader": tr_loader, "tr_dset": tr_dset}


def init_ts_data(data_dir, split, **kwargs):
    ts_dset = PhotoTourismDataset(
        data_dir, split=split, batch_size=None,
        contraction=kwargs['contract'], ndc=kwargs['ndc'],
        scene_bbox=kwargs['scene_bbox'], global_scale=kwargs.get('global_scale', None),
        global_translation=kwargs.get('global_translation', None),
    )
    return {"ts_dset": ts_dset}


def load_data(data_downsample, data_dirs, validate_only, render_only, **kwargs):
    assert len(data_dirs) == 1
    od: Dict[str, Any] = {}
    if not validate_only and not render_only:
        od.update(init_tr_data(data_downsample, data_dirs[0], **kwargs))
    else:
        od.update(tr_loader=None, tr_dset=None)
    test_split = 'render' if render_only else 'test'
    od.update(init_ts_data(data_dirs[0], split=test_split, **kwargs))
    return od
