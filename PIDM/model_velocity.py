#!/usr/bin/env python3
"""Physics-informed conditional DDPM for 2-D NSE velocity reconstruction.

Task:
  8x8 sparse velocity observations -> full 64x64 velocity field.

Target data:
  Data/NSE_Velocity_X_64_centered_pde_dt0.001.npy
  Data/NSE_Velocity_Y_64_centered_pde_dt0.001.npy

The model predicts two channels, (u_x, u_y). The physics term is the continuous
velocity-form Navier-Stokes residual used in calc_nse_velocity_residual.py:

  P[u_t + u . grad(u) - nu Laplacian(u) - f],

where P is the Leray projection that removes the pressure-gradient component.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_beta_schedule(T: int, schedule: str = "cosine") -> torch.Tensor:
    if schedule == "linear":
        return torch.linspace(1e-4, 2e-2, T)
    if schedule != "cosine":
        raise ValueError(f"Unknown beta schedule: {schedule}")

    steps = T + 1
    s = 0.008
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 1e-5, 0.999)


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
    out = a.gather(0, t)
    return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))


def physics_mlflow_params(cfg: "DiffusionConfig") -> Dict[str, str]:
    return {
        "physics_equation": "velocity form, pressure removed by Leray projection",
        "physics_residual": "P[u_t + u.grad(u) - nu*laplacian(u) - f]",
        "physics_grid_size": "64",
        "physics_dx": str(1.0 / 64.0),
        "physics_dy": str(1.0 / 64.0),
        "physics_grid_extent": str(cfg.grid_extent),
        "physics_dt_phys": str(cfg.dt_phys),
        "physics_viscosity": str(cfg.viscosity),
        "physics_derivative_convention": "simulator FFT-index units",
        "physics_forcing": "sampled f=[100*sin(8*pi*y), 0]^T",
    }


class VelocitySparseDataset(Dataset):
    """Consecutive velocity triples with sparse 8x8 velocity observations."""

    def __init__(
        self,
        velocity_x: torch.Tensor,
        velocity_y: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        sensor_stride: int = 8,
    ) -> None:
        if velocity_x.shape != velocity_y.shape:
            raise ValueError(f"Velocity shapes differ: {velocity_x.shape} vs {velocity_y.shape}")
        if velocity_x.ndim != 3 or velocity_x.shape[1:] != (64, 64):
            raise ValueError(f"Expected velocity arrays with shape (N,64,64), got {velocity_x.shape}")

        self.x = torch.stack((velocity_x.float(), velocity_y.float()), dim=1)
        self.mean = mean.float().reshape(1, 2, 1, 1)
        self.std = std.float().reshape(1, 2, 1, 1)
        self.sensor_stride = int(sensor_stride)

    def __len__(self) -> int:
        return max(0, self.x.shape[0] - 2)

    def _normalize(self, field: torch.Tensor) -> torch.Tensor:
        return (field - self.mean.squeeze(0)) / (self.std.squeeze(0) + 1e-8)

    def _sparse_cond(self, field_norm: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros_like(field_norm[:1])
        mask[:, :: self.sensor_stride, :: self.sensor_stride] = 1.0
        sparse_velocity = field_norm * mask
        return torch.cat((sparse_velocity, mask), dim=0)

    def __getitem__(self, idx: int):
        k = idx + 1
        v_prev = self._normalize(self.x[k - 1])
        v_now = self._normalize(self.x[k])
        v_next = self._normalize(self.x[k + 1])
        return (
            v_prev,
            v_now,
            v_next,
            self._sparse_cond(v_prev),
            self._sparse_cond(v_now),
            self._sparse_cond(v_next),
        )


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device).float() / max(half - 1, 1)
        )
        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat((torch.sin(args), torch.cos(args)), dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb


class ResBlockFiLM(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(emb_dim, out_ch * 2)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        scale, shift = self.emb(emb).chunk(2, dim=1)
        h = self.norm2(h)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(self.act(h))
        return h + self.skip(x)


class UNet64FiLM(nn.Module):
    def __init__(self, in_channels: int = 5, out_channels: int = 2, base_ch: int = 64, emb_dim: int = 256):
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, base_ch, 3, padding=1)

        self.rb1 = ResBlockFiLM(base_ch, base_ch, emb_dim)
        self.down1 = nn.Conv2d(base_ch, base_ch, 4, stride=2, padding=1)
        self.rb2 = ResBlockFiLM(base_ch, base_ch * 2, emb_dim)
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 2, 4, stride=2, padding=1)
        self.rb3 = ResBlockFiLM(base_ch * 2, base_ch * 4, emb_dim)
        self.down3 = nn.Conv2d(base_ch * 4, base_ch * 4, 4, stride=2, padding=1)

        self.rb_mid1 = ResBlockFiLM(base_ch * 4, base_ch * 4, emb_dim)
        self.rb_mid2 = ResBlockFiLM(base_ch * 4, base_ch * 4, emb_dim)

        self.up3 = nn.ConvTranspose2d(base_ch * 4, base_ch * 4, 4, stride=2, padding=1)
        self.rb_up3 = ResBlockFiLM(base_ch * 8, base_ch * 2, emb_dim)
        self.up2 = nn.ConvTranspose2d(base_ch * 2, base_ch * 2, 4, stride=2, padding=1)
        self.rb_up2 = ResBlockFiLM(base_ch * 4, base_ch, emb_dim)
        self.up1 = nn.ConvTranspose2d(base_ch, base_ch, 4, stride=2, padding=1)
        self.rb_up1 = ResBlockFiLM(base_ch * 2, base_ch, emb_dim)

        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_conv = nn.Conv2d(base_ch, out_channels, 3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        x = self.in_conv(x)
        h1 = self.rb1(x, emb)
        d1 = self.down1(h1)
        h2 = self.rb2(d1, emb)
        d2 = self.down2(h2)
        h3 = self.rb3(d2, emb)
        d3 = self.down3(h3)
        mid = self.rb_mid2(self.rb_mid1(d3, emb), emb)
        u3 = self.rb_up3(torch.cat((self.up3(mid), h3), dim=1), emb)
        u2 = self.rb_up2(torch.cat((self.up2(u3), h2), dim=1), emb)
        u1 = self.rb_up1(torch.cat((self.up1(u2), h1), dim=1), emb)
        return self.out_conv(self.act(self.out_norm(u1)))


class ConditionalVelocityDDPM(nn.Module):
    def __init__(self, T: int = 1000, emb_dim: int = 256, base_ch: int = 64):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.SiLU(),
            nn.Linear(emb_dim * 4, emb_dim),
        )
        self.unet = UNet64FiLM(in_channels=5, out_channels=2, base_ch=base_ch, emb_dim=emb_dim)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        emb = self.time_mlp(self.time_emb(t))
        if cond is None:
            cond_spatial = torch.zeros(
                x_t.shape[0], 3, x_t.shape[2], x_t.shape[3],
                device=x_t.device, dtype=x_t.dtype
            )
        else:
            cond_spatial = cond
        return self.unet(torch.cat((x_t, cond_spatial), dim=1), emb)


@dataclass
class DiffusionConfig:
    T: int = 1000
    beta_schedule: str = "cosine"
    drop_prob: float = 0.0
    lr: float = 2e-4
    batch_size: int = 32
    num_workers: int = 0
    grad_clip: float = 1.0
    epochs: int = 30
    guidance_scale: float = 1.0
    use_amp: bool = True
    lambda_phys_start: float = 1e-13
    lambda_phys_max: float = 1e-5
    lambda_phys_warmup_ratio: float = 0.5
    physics_weight_cap: float = 5.0
    phys_t_max: int = 100
    dt_phys: float = 0.001
    viscosity: float = 1e-4
    source_grid_size: int = 64
    crop_start: int = 0
    crop_stride: int = 1
    grid_extent: float = 63.0 / 64.0
    forcing_scale: float = 100.0
    forcing_wavenumber: float = 8.0


class DDPMVelocityTrainer:
    def __init__(
        self,
        model: ConditionalVelocityDDPM,
        cfg: DiffusionConfig,
        device: torch.device,
        data_mean: torch.Tensor,
        data_std: torch.Tensor,
    ):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.data_mean = data_mean.to(device).float().reshape(1, 2, 1, 1)
        self.data_std = data_std.to(device).float().reshape(1, 2, 1, 1)

        betas = make_beta_schedule(cfg.T, cfg.beta_schedule).to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.posterior_variance = betas * torch.cat(
            (torch.ones(1, device=device), 1.0 - alphas_cumprod[:-1])
        ) / (1.0 - alphas_cumprod)

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and device.type == "cuda"))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
        )

    @staticmethod
    def _wavenumbers(nx: int, ny: int, device: torch.device, dtype: torch.dtype):
        kx_1d = torch.fft.fftfreq(nx, d=1.0 / nx, device=device)
        ky_1d = torch.fft.fftfreq(ny, d=1.0 / ny, device=device)
        kx = kx_1d.to(dtype).reshape(1, nx, 1)
        ky = ky_1d.to(dtype).reshape(1, 1, ny)
        k2 = kx**2 + ky**2
        k2[:, 0, 0] = 1.0
        return kx, ky, k2

    def gradient(self, field: torch.Tensor):
        kx, ky, _ = self._wavenumbers(field.shape[1], field.shape[2], field.device, field.dtype)
        field_hat = torch.fft.fft2(field)
        return (
            torch.fft.ifft2(1j * kx * field_hat).real,
            torch.fft.ifft2(1j * ky * field_hat).real,
        )

    def laplacian(self, field: torch.Tensor):
        _, _, k2 = self._wavenumbers(field.shape[1], field.shape[2], field.device, field.dtype)
        return torch.fft.ifft2(-k2 * torch.fft.fft2(field)).real

    def forcing(self, batch_size: int, nx: int, ny: int, device: torch.device, dtype: torch.dtype):
        y_indices = self.cfg.crop_start + self.cfg.crop_stride * torch.arange(ny, device=device, dtype=dtype)
        y = self.cfg.grid_extent * y_indices / (self.cfg.source_grid_size - 1)
        force_x = self.cfg.forcing_scale * torch.sin(self.cfg.forcing_wavenumber * torch.pi * y)
        force_x = force_x.reshape(1, 1, ny).expand(batch_size, nx, ny)
        return force_x, torch.zeros_like(force_x)

    def project_divergence_free(self, qx: torch.Tensor, qy: torch.Tensor):
        kx, ky, k2 = self._wavenumbers(qx.shape[1], qx.shape[2], qx.device, qx.dtype)
        qx_hat = torch.fft.fft2(qx)
        qy_hat = torch.fft.fft2(qy)
        div_hat = 1j * kx * qx_hat + 1j * ky * qy_hat
        pressure_hat = div_hat / k2
        pressure_hat[:, 0, 0] = 0.0
        residual_x_hat = qx_hat + 1j * kx * pressure_hat
        residual_y_hat = qy_hat + 1j * ky * pressure_hat
        return torch.fft.ifft2(residual_x_hat).real, torch.fft.ifft2(residual_y_hat).real

    def velocity_residual(self, vel_prev: torch.Tensor, vel_now: torch.Tensor, vel_next: torch.Tensor):
        u_prev, v_prev = vel_prev[:, 0], vel_prev[:, 1]
        u_now, v_now = vel_now[:, 0], vel_now[:, 1]
        u_next, v_next = vel_next[:, 0], vel_next[:, 1]

        u_t = (u_next - u_prev) / (2.0 * self.cfg.dt_phys)
        v_t = (v_next - v_prev) / (2.0 * self.cfg.dt_phys)
        u_x, u_y = self.gradient(u_now)
        v_x, v_y = self.gradient(v_now)
        lap_u = self.laplacian(u_now)
        lap_v = self.laplacian(v_now)
        force_x, force_y = self.forcing(vel_now.shape[0], vel_now.shape[2], vel_now.shape[3], vel_now.device, vel_now.dtype)

        adv_x = u_now * u_x + v_now * u_y
        adv_y = u_now * v_x + v_now * v_y
        qx = u_t + adv_x - self.cfg.viscosity * lap_u - force_x
        qy = v_t + adv_y - self.cfg.viscosity * lap_v - force_y
        residual_x, residual_y = self.project_divergence_free(qx, qy)
        residual = torch.stack((residual_x, residual_y), dim=1)
        terms = {
            "time_derivative": torch.mean(u_t**2 + v_t**2).detach(),
            "advection": torch.mean(adv_x**2 + adv_y**2).detach(),
            "diffusion": torch.mean((-self.cfg.viscosity * lap_u) ** 2 + (-self.cfg.viscosity * lap_v) ** 2).detach(),
            "forcing": torch.mean(force_x**2 + force_y**2).detach(),
            "residual_abs_mean": residual.abs().mean().detach(),
            "residual_rmse": torch.sqrt(torch.mean(residual**2)).detach(),
        }
        return residual, terms

    @staticmethod
    def get_lambda_phys(epoch: int, total_epochs: int, lambda_start: float, lambda_max: float, warmup_ratio: float) -> float:
        warmup_epochs = max(1, int(total_epochs * warmup_ratio))
        progress = min(epoch / warmup_epochs, 1.0)
        return lambda_start + progress**2 * (lambda_max - lambda_start)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.data_std + 1e-8) + self.data_mean

    def predict_x0(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        eps_pred = self.model(x_t, t, cond)
        sqrt_acp = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_om = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return (x_t - sqrt_om * eps_pred) / (sqrt_acp + 1e-8)

    def train_one_epoch(self, loader: DataLoader, epoch: int, global_step: int):
        self.model.train()
        total_loss = total_data = total_phys = total_weighted_phys = 0.0
        n_seen = 0
        lambda_phys = self.get_lambda_phys(
            epoch,
            self.cfg.epochs,
            self.cfg.lambda_phys_start,
            self.cfg.lambda_phys_max,
            self.cfg.lambda_phys_warmup_ratio,
        )

        for batch_idx, (vel_prev, vel_now, vel_next, cond_prev, cond_now, cond_next) in enumerate(loader, start=1):
            vel_prev = vel_prev.to(self.device)
            vel_now = vel_now.to(self.device)
            vel_next = vel_next.to(self.device)
            cond_prev = cond_prev.to(self.device)
            cond_now = cond_now.to(self.device)
            cond_next = cond_next.to(self.device)

            batch_size = vel_now.shape[0]
            t = torch.randint(0, self.cfg.T, (batch_size,), device=self.device, dtype=torch.long)
            noise = torch.randn_like(vel_now)
            x_t = self.q_sample(vel_now, t, noise)
            cond_mask = torch.rand(batch_size, device=self.device) > self.cfg.drop_prob

            self.opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(self.cfg.use_amp and self.device.type == "cuda")):
                idx_c = torch.nonzero(cond_mask, as_tuple=False).squeeze(1)
                idx_u = torch.nonzero(~cond_mask, as_tuple=False).squeeze(1)
                loss_sum = x_t.new_tensor(0.0)
                denom = 0

                if idx_c.numel() > 0:
                    eps_pred_c = self.model(x_t[idx_c], t[idx_c], cond_now[idx_c])
                    loss_sum = loss_sum + F.mse_loss(eps_pred_c, noise[idx_c]) * idx_c.numel()
                    denom += idx_c.numel()
                if idx_u.numel() > 0:
                    eps_pred_u = self.model(x_t[idx_u], t[idx_u], None)
                    loss_sum = loss_sum + F.mse_loss(eps_pred_u, noise[idx_u]) * idx_u.numel()
                    denom += idx_u.numel()

                loss_data = loss_sum / max(denom, 1)
                idx_phys = idx_c[t[idx_c] < self.cfg.phys_t_max] if idx_c.numel() > 0 else idx_c
                if idx_phys.numel() > 0:
                    t_p = t[idx_phys]
                    x0_pred = self.predict_x0(x_t[idx_phys], t_p, cond_now[idx_phys])

                    noise_prev = torch.randn_like(vel_prev[idx_phys])
                    noise_next = torch.randn_like(vel_next[idx_phys])
                    x_t_prev = self.q_sample(vel_prev[idx_phys], t_p, noise_prev)
                    x_t_next = self.q_sample(vel_next[idx_phys], t_p, noise_next)
                    prev_pred = self.predict_x0(x_t_prev, t_p, cond_prev[idx_phys])
                    next_pred = self.predict_x0(x_t_next, t_p, cond_next[idx_phys])

                    residual, terms = self.velocity_residual(
                        self.denormalize(prev_pred),
                        self.denormalize(x0_pred),
                        self.denormalize(next_pred),
                    )
                    loss_phys = torch.mean(residual**2)
                else:
                    loss_phys = x_t.new_tensor(0.0)
                    terms = {}

                weighted_phys_raw = lambda_phys * loss_phys
                weighted_phys = torch.clamp(weighted_phys_raw, max=self.cfg.physics_weight_cap)
                loss = loss_data + weighted_phys

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Skipping bad batch {batch_idx}: data={loss_data.item()} phys={loss_phys.item()}")
                continue

            self.scaler.scale(loss).backward()
            if self.cfg.grad_clip is not None:
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.scaler.step(self.opt)
            self.scaler.update()

            global_step += 1
            total_loss += float(loss.item()) * batch_size
            total_data += float(loss_data.item()) * batch_size
            total_phys += float(loss_phys.item()) * batch_size
            total_weighted_phys += float(weighted_phys.item()) * batch_size
            n_seen += batch_size

            mlflow.log_metric("step_loss", float(loss.item()), step=global_step)
            mlflow.log_metric("step_loss_data", float(loss_data.item()), step=global_step)
            mlflow.log_metric("step_loss_physics", float(loss_phys.item()), step=global_step)
            mlflow.log_metric("step_loss_physics_weighted", float(weighted_phys.item()), step=global_step)
            mlflow.log_metric("lambda_phys", float(lambda_phys), step=global_step)

            if batch_idx == 1 and terms:
                for key, value in terms.items():
                    mlflow.log_metric(f"first_batch_{key}", float(value.cpu()), step=epoch)
                    print(f"  first batch {key}: {float(value.cpu()):.6e}")

            print(
                f"epoch={epoch:03d} batch={batch_idx:04d}/{len(loader)} "
                f"loss={loss.item():.6e} data={loss_data.item():.6e} "
                f"phys={loss_phys.item():.6e} weighted_phys={weighted_phys.item():.6e} "
                f"lambda={lambda_phys:.2e}"
            )

        denom = max(n_seen, 1)
        return {
            "loss": total_loss / denom,
            "data": total_data / denom,
            "physics": total_phys / denom,
            "physics_weighted": total_weighted_phys / denom,
            "global_step": global_step,
        }

    @torch.no_grad()
    def sample_cfg(self, cond: torch.Tensor, guidance_scale: float, shape: Tuple[int, int, int, int]) -> torch.Tensor:
        self.model.eval()
        x = torch.randn(shape, device=self.device)
        for i in reversed(range(self.cfg.T)):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            eps_c = self.model(x, t, cond)
            eps_u = self.model(x, t, None)
            eps = eps_u + guidance_scale * (eps_c - eps_u)
            sqrt_acp = extract(self.sqrt_alphas_cumprod, t, x.shape)
            sqrt_om = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
            x0_pred = (x - sqrt_om * eps) / (sqrt_acp + 1e-8)

            betas_t = extract(self.betas, t, x.shape)
            alphas_t = extract(self.alphas, t, x.shape)
            acp_t = extract(self.alphas_cumprod, t, x.shape)
            acp_prev = torch.cat((torch.ones(1, device=self.device), self.alphas_cumprod[:-1]), dim=0)
            acp_prev_t = extract(acp_prev, t, x.shape)
            coef1 = betas_t * torch.sqrt(acp_prev_t) / (1.0 - acp_t + 1e-8)
            coef2 = (1.0 - acp_prev_t) * torch.sqrt(alphas_t) / (1.0 - acp_t + 1e-8)
            mean = coef1 * x0_pred + coef2 * x
            if i > 0:
                x = mean + torch.sqrt(extract(self.posterior_variance, t, x.shape) + 1e-8) * torch.randn_like(x)
            else:
                x = mean
        return x

    @torch.no_grad()
    def eval_recon_mse(self, loader: DataLoader, num_batches: int = 2) -> float:
        self.model.eval()
        mses = []
        for batch_idx, (_, vel_now, _, _, cond_now, _) in enumerate(loader):
            if batch_idx >= num_batches:
                break
            vel_now = vel_now.to(self.device)
            cond_now = cond_now.to(self.device)
            pred = self.sample_cfg(cond_now, self.cfg.guidance_scale, vel_now.shape)
            mses.append(F.mse_loss(pred, vel_now).item())
        return float(np.mean(mses)) if mses else float("nan")


def load_velocity_arrays(path_x: Path, path_y: Path):
    vx = np.load(path_x, mmap_mode="r")
    vy = np.load(path_y, mmap_mode="r")
    if vx.shape != vy.shape:
        raise ValueError(f"Velocity component shapes differ: {vx.shape} vs {vy.shape}")
    return torch.from_numpy(np.asarray(vx, dtype=np.float32)), torch.from_numpy(np.asarray(vy, dtype=np.float32))


def run_training(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device(args.device) if args.device else default_device()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vx, vy = load_velocity_arrays(args.velocity_x_dataset, args.velocity_y_dataset)
    n_total = vx.shape[0]
    n_train = int(0.8 * n_total)
    train_x, test_x = vx[:n_train], vx[n_train:]
    train_y, test_y = vy[:n_train], vy[n_train:]

    mean = torch.tensor(
        [train_x.float().mean().item(), train_y.float().mean().item()],
        dtype=torch.float32,
    )
    std = torch.tensor(
        [train_x.float().std().item(), train_y.float().std().item()],
        dtype=torch.float32,
    )
    print(f"Train mean ux/uy: {mean.tolist()}")
    print(f"Train std ux/uy: {std.tolist()}")

    train_ds = VelocitySparseDataset(train_x, train_y, mean, std, args.sensor_stride)
    test_ds = VelocitySparseDataset(test_x, test_y, mean, std, args.sensor_stride)
    cfg = DiffusionConfig(
        T=args.T,
        beta_schedule=args.beta_schedule,
        drop_prob=args.drop_prob,
        lr=args.lr,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        grad_clip=args.grad_clip,
        epochs=args.epochs,
        guidance_scale=args.guidance_scale,
        use_amp=not args.no_amp,
        lambda_phys_start=args.lambda_phys_start,
        lambda_phys_max=args.lambda_phys_max,
        lambda_phys_warmup_ratio=args.lambda_phys_warmup_ratio,
        physics_weight_cap=args.physics_weight_cap,
        phys_t_max=args.phys_t_max,
        dt_phys=args.dt_phys,
        viscosity=args.viscosity,
        source_grid_size=args.source_grid_size,
        crop_start=args.crop_start,
        crop_stride=args.crop_stride,
        grid_extent=args.grid_extent,
        forcing_scale=args.forcing_scale,
        forcing_wavenumber=args.forcing_wavenumber,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    model = ConditionalVelocityDDPM(T=cfg.T, emb_dim=args.emb_dim, base_ch=args.base_ch)
    trainer = DDPMVelocityTrainer(model, cfg, device, mean, std)

    mlflow.set_tracking_uri(f"file:{out_dir / 'mlruns'}")
    mlflow.set_experiment(args.mlflow_experiment)

    global_step = 0
    best_test = float("inf")
    with mlflow.start_run(run_name=args.mlflow_run_name):
        mlflow.log_params(cfg.__dict__)
        mlflow.log_params(physics_mlflow_params(cfg))
        mlflow.log_param("velocity_x_dataset", str(args.velocity_x_dataset))
        mlflow.log_param("velocity_y_dataset", str(args.velocity_y_dataset))
        mlflow.log_param("seed", args.seed)
        mlflow.log_param("train_mean_ux", float(mean[0]))
        mlflow.log_param("train_mean_uy", float(mean[1]))
        mlflow.log_param("train_std_ux", float(std[0]))
        mlflow.log_param("train_std_uy", float(std[1]))
        mlflow.log_param("train_samples", len(train_ds))
        mlflow.log_param("test_samples", len(test_ds))

        for epoch in range(1, cfg.epochs + 1):
            start = time.time()
            metrics = trainer.train_one_epoch(train_loader, epoch, global_step)
            global_step = int(metrics["global_step"])
            test_mse = trainer.eval_recon_mse(test_loader, num_batches=args.eval_batches)
            elapsed = time.time() - start

            mlflow.log_metric("train_loss", metrics["loss"], step=epoch)
            mlflow.log_metric("train_loss_data", metrics["data"], step=epoch)
            mlflow.log_metric("train_loss_physics", metrics["physics"], step=epoch)
            mlflow.log_metric("train_loss_physics_weighted", metrics["physics_weighted"], step=epoch)
            mlflow.log_metric("test_recon_mse", test_mse, step=epoch)

            ckpt = {
                "model": trainer.model.state_dict(),
                "cfg": cfg.__dict__,
                "train_mean": mean,
                "train_std": std,
                "last_epoch": epoch,
                "global_step": global_step,
                "optimizer": trainer.opt.state_dict(),
                "scaler": trainer.scaler.state_dict(),
                "best_test_recon_mse": best_test,
            }
            torch.save(ckpt, out_dir / "conditional.pt")

            if epoch == 1 or test_mse < best_test:
                best_test = test_mse
                ckpt["best_test_recon_mse"] = best_test
                torch.save(ckpt, out_dir / "best.pt")

            print(
                f"Epoch {epoch:03d} | train_loss={metrics['loss']:.6e} "
                f"data={metrics['data']:.6e} phys={metrics['physics']:.6e} "
                f"weighted_phys={metrics['physics_weighted']:.6e} "
                f"test_mse={test_mse:.6e} | {elapsed:.1f}s"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train velocity PIDM for sparse 8x8 -> full 64x64 reconstruction.")
    parser.add_argument("--velocity-x-dataset", type=Path, default=Path("/content/drive/MyDrive/Lab/CondDiff/NSE_Velocity_X.npy"))
    parser.add_argument("--velocity-y-dataset", type=Path, default=Path("/content/drive/MyDrive/Lab/CondDiff/NSE_Velocity_Y.npy"))
    parser.add_argument("--out-dir", type=Path, default=Path("/content/drive/MyDrive/Lab/CondDiff"))
    parser.add_argument("--mlflow-experiment", default="pidm_2dns_velocity")
    parser.add_argument("--mlflow-run-name", default="velocity_sparse_conditional_ddpm")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--beta-schedule", choices=("cosine", "linear"), default="cosine")
    parser.add_argument("--drop-prob", type=float, default=0.0)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--base-ch", type=int, default=64)
    parser.add_argument("--emb-dim", type=int, default=256)
    parser.add_argument("--sensor-stride", type=int, default=8)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--eval-batches", type=int, default=2)
    parser.add_argument("--lambda-phys-start", type=float, default=1e-15)
    parser.add_argument("--lambda-phys-max", type=float, default=1e-10)
    parser.add_argument("--lambda-phys-warmup-ratio", type=float, default=0.5)
    parser.add_argument("--physics-weight-cap", type=float, default=5.0)
    parser.add_argument("--phys-t-max", type=int, default=100)
    parser.add_argument("--dt-phys", type=float, default=0.001)
    parser.add_argument("--viscosity", type=float, default=1.0e-4)
    parser.add_argument("--source-grid-size", type=int, default=64)
    parser.add_argument("--crop-start", type=int, default=0)
    parser.add_argument("--crop-stride", type=int, default=1)
    parser.add_argument("--grid-extent", type=float, default=63.0 / 64.0)
    parser.add_argument("--forcing-scale", type=float, default=100.0)
    parser.add_argument("--forcing-wavenumber", type=float, default=8.0)
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
