# ============================================
# PIDM: Physics-Informed Conditional DDPM (Sparse 8x8 -> Full 64x64)
# Classifier-Free Guidance (CFG) in PyTorch
#
# Assumptions:
# - `data` is a NumPy array or torch Tensor of shape (N, 64, 64) (vorticity).
#
# What this does:
# - Splits first 80% train, last 20% test.
# - Sparse 8x8 observations on a fixed grid (stride=8) + binary mask condition.
# - Trains conditional DDPM with optional CFG dropout.
# - Physics loss: predict vorticity at k-1, k, k+1 and penalize the 2D NS
#   vorticity residual (central time derivative + spectral advection/diffusion).
#
# Notes:
# - U-Net input: cat([x_t, sparse_field, mask], dim=1) -> 3 channels.
# - FiLM modulation uses only the diffusion timestep embedding.
# ============================================

import math
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import mlflow
from mlflow.tracking import MlflowClient


# -------------------------
# Utilities
# -------------------------
def seed_everything(seed: int = 0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_beta_schedule(T: int, kind: str = "cosine") -> torch.Tensor:
    """
    Returns betas of shape (T,) in float32.
    Cosine schedule per Nichol & Dhariwal (common stable default).
    """
    if kind == "linear":
        beta_start = 1e-4
        beta_end = 2e-2
        return torch.linspace(beta_start, beta_end, T, dtype=torch.float32)

    if kind != "cosine":
        raise ValueError(f"Unknown schedule: {kind}")

    # cosine
    s = 0.008
    steps = T + 1
    x = torch.linspace(0, T, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clip(betas, 1e-8, 0.999).float()
    return betas


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """
    Extract values from 1-D tensor a at indices t and reshape to [B, 1, 1, 1] for broadcast.
    """
    b = t.shape[0]
    out = a.gather(0, t).reshape(b, *((1,) * (len(x_shape) - 1)))
    return out


def physics_mlflow_params(grid_size: int = 64) -> Dict[str, str]:
  """Hardcoded PDE domain/forcing settings for MLflow (must match pde_residual)."""
  dx = 1.0 / grid_size
  return {
      "physics_domain": "[0,1]^2 periodic",
      "physics_grid_size": str(grid_size),
      "physics_dx": str(dx),
      "physics_dy": str(dx),
      "physics_forcing": "f=[100*sin(8y), 0]^T",
      "physics_forcing_curl_z": "-800*cos(8y)",
  }


def log_physics_mlflow_params(grid_size: int = 64) -> None:
    mlflow.log_params(physics_mlflow_params(grid_size))


# -------------------------
# Dataset: fixed split, full 64x64 field used as both target x0 and condition y
# -------------------------
class NavierStokesSparseDataset(Dataset):
    def __init__(
        self,
        full_fields: torch.Tensor,   # (N, 64, 64)
        mean: float,
        std: float,
        sensor_stride: int = 8,
    ):
        """
        sensor_stride=8 on a 64x64 grid gives an 8x8 observation lattice
        (i.e. 64 observed points total).

        Returns consecutive triplets (k-1, k, k+1) for PIDM physics loss.
        """
        assert full_fields.ndim == 3 and full_fields.shape[1:] == (64, 64)
        self.x = full_fields.float()
        self.mean = float(mean)
        self.std = float(std)
        self.sensor_stride = int(sensor_stride)

    def __len__(self):
        # centers k must have both temporal neighbors
        return max(0, self.x.shape[0] - 2)

    def _normalize(self, field: torch.Tensor) -> torch.Tensor:
        return (field - self.mean) / (self.std + 1e-8)

    def _sparse_cond(self, field_norm: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros_like(field_norm)
        mask[::self.sensor_stride, ::self.sensor_stride] = 1.0
        y_sparse = field_norm * mask
        return torch.stack([y_sparse, mask], dim=0)  # (2, 64, 64)

    def __getitem__(self, idx: int):
        k = idx + 1
        x_prev = self._normalize(self.x[k - 1])
        x0 = self._normalize(self.x[k])
        x_next = self._normalize(self.x[k + 1])

        cond_prev = self._sparse_cond(x_prev)
        cond = self._sparse_cond(x0)
        cond_next = self._sparse_cond(x_next)

        return (
            x_prev.unsqueeze(0),
            x0.unsqueeze(0),
            x_next.unsqueeze(0),
            cond_prev,
            cond,
            cond_next,
        )

# -------------------------
# Time embedding
# -------------------------
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (B,) int64 or float
        returns: (B, dim)
        """
        half = self.dim // 2
        device = t.device
        t = t.float()
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half, device=device).float() / (half - 1)
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


# -------------------------
# ResBlock with FiLM (AdaGN-like) conditioning
# -------------------------
class ResBlockFiLM(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, groups: int = 8):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.emb_proj = nn.Linear(emb_dim, 2 * out_ch)  # gamma, beta
        self.act = nn.SiLU()

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        emb: (B, emb_dim)
        """
        h = self.conv1(self.act(self.norm1(x)))

        # FiLM on second norm
        gamma_beta = self.emb_proj(emb)  # (B, 2*out_ch)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]

        h = self.norm2(h)
        h = h * (1 + gamma) + beta
        h = self.conv2(self.act(h))

        return h + self.skip(x)


# -------------------------
# Simple U-Net backbone (64x64)
# -------------------------
class UNet64FiLM(nn.Module):
    def __init__(self, base_ch: int = 64, emb_dim: int = 256):
        super().__init__()
        self.in_conv = nn.Conv2d(3, base_ch, 3, padding=1)

        # Down
        self.rb1 = ResBlockFiLM(base_ch, base_ch, emb_dim)
        self.down1 = nn.Conv2d(base_ch, base_ch, 4, stride=2, padding=1)  # 64->32

        self.rb2 = ResBlockFiLM(base_ch, base_ch * 2, emb_dim)
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 2, 4, stride=2, padding=1)  # 32->16

        self.rb3 = ResBlockFiLM(base_ch * 2, base_ch * 4, emb_dim)
        self.down3 = nn.Conv2d(base_ch * 4, base_ch * 4, 4, stride=2, padding=1)  # 16->8

        # Bottleneck
        self.rb_mid1 = ResBlockFiLM(base_ch * 4, base_ch * 4, emb_dim)
        self.rb_mid2 = ResBlockFiLM(base_ch * 4, base_ch * 4, emb_dim)

        # Up
        self.up3 = nn.ConvTranspose2d(base_ch * 4, base_ch * 4, 4, stride=2, padding=1)  # 8->16
        self.rb_up3 = ResBlockFiLM(base_ch * 8, base_ch * 2, emb_dim)

        self.up2 = nn.ConvTranspose2d(base_ch * 2, base_ch * 2, 4, stride=2, padding=1)  # 16->32
        self.rb_up2 = ResBlockFiLM(base_ch * 4, base_ch, emb_dim)

        self.up1 = nn.ConvTranspose2d(base_ch, base_ch, 4, stride=2, padding=1)  # 32->64
        self.rb_up1 = ResBlockFiLM(base_ch * 2, base_ch, emb_dim)

        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_conv = nn.Conv2d(base_ch, 1, 3, padding=1)

        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        # x: (B,3,64,64)
        # channel 0 = x_t
        # channel 1 = sparse observed field
        # channel 2 = binary mask
        x = self.in_conv(x)

        h1 = self.rb1(x, emb)          # (B, base, 64,64)
        d1 = self.down1(h1)            # (B, base, 32,32)

        h2 = self.rb2(d1, emb)         # (B, 2base, 32,32)
        d2 = self.down2(h2)            # (B, 2base, 16,16)

        h3 = self.rb3(d2, emb)         # (B, 4base, 16,16)
        d3 = self.down3(h3)            # (B, 4base, 8,8)

        mid = self.rb_mid1(d3, emb)
        mid = self.rb_mid2(mid, emb)

        u3 = self.up3(mid)             # (B, 4base, 16,16)
        u3 = torch.cat([u3, h3], dim=1)
        u3 = self.rb_up3(u3, emb)      # (B, 2base, 16,16)

        u2 = self.up2(u3)              # (B, 2base, 32,32)
        u2 = torch.cat([u2, h2], dim=1)
        u2 = self.rb_up2(u2, emb)      # (B, base, 32,32)

        u1 = self.up1(u2)              # (B, base, 64,64)
        u1 = torch.cat([u1, h1], dim=1)
        u1 = self.rb_up1(u1, emb)      # (B, base, 64,64)

        out = self.out_conv(self.act(self.out_norm(u1)))
        return out  # predicted noise eps


# -------------------------
# Full Conditional DDPM Model (CFG-ready)
# -------------------------
class ConditionalDDPM(nn.Module):
    def __init__(self, T: int = 1000, emb_dim: int = 256, base_ch: int = 64):
        super().__init__()
        self.T = T

        self.time_emb = SinusoidalTimeEmbedding(emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.SiLU(),
            nn.Linear(emb_dim * 4, emb_dim),
        )

        self.unet = UNet64FiLM(base_ch=base_ch, emb_dim=emb_dim)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        """
        x_t:  (B,1,64,64)
        t:    (B,) int64
        cond: (B,2,64,64) or None
            cond[:,0] = sparse observed field
            cond[:,1] = binary mask
        """
        emb = self.time_mlp(self.time_emb(t))

        B, _, H, W = x_t.shape

        if cond is None:
            cond_spatial = torch.zeros(B, 2, H, W, device=x_t.device, dtype=x_t.dtype)
        else:
            cond_spatial = cond

        x_in = torch.cat([x_t, cond_spatial], dim=1)  # (B,3,64,64)

        return self.unet(x_in, emb)


# -------------------------
# DDPM Diffusion wrapper (training + sampling)
# -------------------------
@dataclass
class DiffusionConfig:
    T: int = 1000
    beta_schedule: str = "cosine"
    drop_prob: float = 0     # CFG condition dropout probability
    lr: float = 2e-4
    batch_size: int = 64
    num_workers: int = 0  # Set to 0 for macOS compatibility (multiprocessing issues)
    grad_clip: float = 1.0
    epochs: int = 30
    guidance_scale: float = 1.0  # CFG sampling scale
    use_amp: bool = True
    #lambda_phys: float = 5e-9
    lambda_phys_start: float = 5e-11
    lambda_phys_max: float = 5e-9
    lambda_phys_warmup_ratio: float = 0.5
    dt_phys: float = 0.01
    viscosity: float = 1e-4
    # Low-pass cutoff in angular wavenumber |k| for physics loss (None = full spectrum).
    low_freq_k_cutoff: Optional[float] = 2.0


class DDPMTrainer:
    def __init__(
        self,
        model: ConditionalDDPM,
        cfg: DiffusionConfig,
        device: torch.device,
        data_mean: float = 0.0,
        data_std: float = 1.0,
    ):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.data_mean = float(data_mean)
        self.data_std = float(data_std)
        self.dt = cfg.dt_phys

        betas = make_beta_schedule(cfg.T, cfg.beta_schedule).to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        # posterior variance for sampling
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), alphas_cumprod[:-1]], dim=0)
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and device.type == "cuda"))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0 + \
               extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise

    def pde_residual(
        self,
        omega_k_minus_1: torch.Tensor,
        omega_k: torch.Tensor,
        omega_k_plus_1: torch.Tensor,
    ) -> torch.Tensor:
        """
        2D incompressible Navier-Stokes vorticity residual:
            dω/dt + u * dω/dx + v * dω/dy - ν * ∇²ω - (curl f)_z
        with forcing:
            f = [100 sin(8y), 0]^T
        Uses central difference in time and spectral spatial derivatives (FFT).
        Grid spacing on [0,1]^2: dx = dy = 1/64 (for H=W=64).
        """
        dt = self.dt
        nu = self.cfg.viscosity

        w_prev = omega_k_minus_1.squeeze(1)
        w_cur = omega_k.squeeze(1)
        w_next = omega_k_plus_1.squeeze(1)

        B, H, W = w_cur.shape
        device = w_cur.device
        dx = 1.0 / H
        dy = 1.0 / W

        kx = (2 * math.pi) * torch.fft.fftfreq(H, d=dx, device=device).view(H, 1)
        ky = (2 * math.pi) * torch.fft.rfftfreq(W, d=dy, device=device).view(1, W // 2 + 1)

        k2 = kx**2 + ky**2
        k_cutoff = self.cfg.low_freq_k_cutoff
        if k_cutoff is not None and k_cutoff > 0:
            low_mask = (k2 < (float(k_cutoff) ** 2)).to(dtype=torch.float32)
            w_prev = torch.fft.irfft2(torch.fft.rfft2(w_prev) * low_mask, s=(H, W))
            w_cur = torch.fft.irfft2(torch.fft.rfft2(w_cur) * low_mask, s=(H, W))
            w_next = torch.fft.irfft2(torch.fft.rfft2(w_next) * low_mask, s=(H, W))

        w_fft = torch.fft.rfft2(w_cur)
        k2_safe = k2.clone()
        k2_safe[0, 0] = 1.0

        eps = 1e-6
        psi_fft = -w_fft / (k2_safe + eps)
        psi_fft[..., 0, 0] = 0.0

        u = torch.fft.irfft2(1j * ky * psi_fft, s=(H, W))
        v = torch.fft.irfft2(-1j * kx * psi_fft, s=(H, W))
        #u = torch.clamp(u, -10.0, 10.0)
        #v = torch.clamp(v, -10.0, 10.0)

        w_x = torch.fft.irfft2(1j * kx * w_fft, s=(H, W))
        w_y = torch.fft.irfft2(1j * ky * w_fft, s=(H, W))
        #w_x = torch.clamp(w_x, -100.0, 100.0)
        #w_y = torch.clamp(w_y, -100.0, 100.0)

        lap_fft = -(kx**2 + ky**2) * w_fft
        #lap_fft = torch.clamp(lap_fft.real, -1e6, 1e6) + 1j * torch.clamp(lap_fft.imag, -1e6, 1e6)
        lap_w = torch.fft.irfft2(lap_fft, s=(H, W))

        # f = [100 sin(8y), 0]^T => (curl f)_z = -800 cos(8y); y in physical coords [0,1)
        y_phys = torch.arange(W, device=device, dtype=w_cur.dtype) * dy
        forcing_vort = (-800.0 * torch.cos(8.0 * y_phys)).view(1, 1, W).expand(B, H, W)

        w_t = (w_next - w_prev) / (2.0 * dt)
        R = w_t + u * w_x + v * w_y - nu * lap_w - forcing_vort
        return R.unsqueeze(1)

    def p_mean_variance(self, x_t: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns mean and variance of p(x_{t-1} | x_t, y).
        We predict eps, then compute x0_pred and posterior mean.
        """
        eps = self.model(x_t, t, y)

        sqrt_acp = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_om = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)

        x0_pred = (x_t - sqrt_om * eps) / (sqrt_acp + 1e-8)

        # DDPM posterior mean formula:
        betas_t = extract(self.betas, t, x_t.shape)
        alphas_t = extract(self.alphas, t, x_t.shape)

        acp_t = extract(self.alphas_cumprod, t, x_t.shape)
        acp_prev = torch.cat([torch.ones(1, device=self.device), self.alphas_cumprod[:-1]], dim=0)
        acp_prev_t = extract(acp_prev, t, x_t.shape)

        coef1 = betas_t * torch.sqrt(acp_prev_t) / (1.0 - acp_t + 1e-8)
        coef2 = (1.0 - acp_prev_t) * torch.sqrt(alphas_t) / (1.0 - acp_t + 1e-8)

        mean = coef1 * x0_pred + coef2 * x_t
        var = extract(self.posterior_variance, t, x_t.shape)
        return mean, var

    @torch.no_grad()
    def sample_cfg(self, cond: torch.Tensor, guidance_scale: float, shape: Tuple[int, int, int, int]) -> torch.Tensor:
        """
        CFG sampling:
          eps = eps_uncond + s*(eps_cond - eps_uncond)
        y: (B,1,64,64)
        shape: (B,1,64,64)
        returns x0 samples (B,1,64,64)
        """
        self.model.eval()
        x = torch.randn(shape, device=self.device)

        for i in reversed(range(self.cfg.T)):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)

            # unconditional and conditional eps
            # eps_u = self.model(x, t, None)
            # eps_c = self.model(x, t, y)
            # eps = eps_u + guidance_scale * (eps_c - eps_u)
            eps = self.model(x, t, cond)

            # compute mean/var using eps (manual to avoid double forward)
            sqrt_acp = extract(self.sqrt_alphas_cumprod, t, x.shape)
            sqrt_om = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
            x0_pred = (x - sqrt_om * eps) / (sqrt_acp + 1e-8)

            betas_t = extract(self.betas, t, x.shape)
            alphas_t = extract(self.alphas, t, x.shape)
            acp_t = extract(self.alphas_cumprod, t, x.shape)
            acp_prev = torch.cat([torch.ones(1, device=self.device), self.alphas_cumprod[:-1]], dim=0)
            acp_prev_t = extract(acp_prev, t, x.shape)

            coef1 = betas_t * torch.sqrt(acp_prev_t) / (1.0 - acp_t + 1e-8)
            coef2 = (1.0 - acp_prev_t) * torch.sqrt(alphas_t) / (1.0 - acp_t + 1e-8)
            mean = coef1 * x0_pred + coef2 * x

            var = extract(self.posterior_variance, t, x.shape)
            if i > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(var + 1e-8) * noise
            else:
                x = mean

        return x

    def spatial_smoothness_loss(self, w):
        """
        w: (B, 1, H, W)
        Penalizes nonsmooth behavior in x and y directions.
        """
        dx = w[:, :, :, 1:] - w[:, :, :, :-1]
        dy = w[:, :, 1:, :] - w[:, :, :-1, :]
        return (dx ** 2).mean() + (dy ** 2).mean()

    def temporal_second_difference_loss(self, w_prev, w_cur, w_next):
        """
        Penalizes nonsmooth acceleration in time:
        w_{k+1} - 2w_k + w_{k-1}
        """
        dtt = w_next - 2.0 * w_cur + w_prev
        return (dtt ** 2).mean()


    @staticmethod
    def get_lambda_phys(epoch, total_epochs, lambda_start, lambda_max, warmup_ratio=0.5):
        warmup_epochs = max(1, int(total_epochs * warmup_ratio))
        progress = min(epoch / warmup_epochs, 1.0)

        # quadratic ramp
        ramp = progress ** 2

        return lambda_start + ramp * (lambda_max - lambda_start)

    def train_one_epoch(self, loader: DataLoader, epoch: int):
        self.model.train()
        total_loss = 0.0
        total_diff_loss = 0.0
        total_phys_loss = 0.0
        total_smooth_space_loss = 0.0
        total_smooth_time_loss = 0.0
        n = 0
        num_batches = len(loader)
        lambda_smooth_space = 1e-4
        lambda_smooth_time = 5e-5

        print(f"  Starting epoch {epoch} ({num_batches} batches)...")

        for batch_idx, (omega_prev, x0, omega_next, cond_prev, cond, cond_next) in enumerate(loader):
            omega_prev = omega_prev.to(self.device)
            x0 = x0.to(self.device)
            omega_next = omega_next.to(self.device)
            cond_prev = cond_prev.to(self.device)
            cond = cond.to(self.device)
            cond_next = cond_next.to(self.device)

            B = x0.size(0)
            t = torch.randint(0, self.cfg.T, (B,), device=self.device, dtype=torch.long)
            noise = torch.randn_like(x0)
            x_t = self.q_sample(x0, t, noise)

            cond_mask = (torch.rand(B, device=self.device) > self.cfg.drop_prob)
            self.opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(self.cfg.use_amp and self.device.type == "cuda")):
                idx_c = torch.nonzero(cond_mask, as_tuple=False).squeeze(1)
                idx_u = torch.nonzero(~cond_mask, as_tuple=False).squeeze(1)

                loss = 0.0
                denom = 0

                if idx_c.numel() > 0:
                    eps_pred_c = self.model(x_t[idx_c], t[idx_c], cond[idx_c])
                    loss_c = F.mse_loss(eps_pred_c, noise[idx_c])
                    loss = loss + loss_c * idx_c.numel()
                    denom += idx_c.numel()

                if idx_u.numel() > 0:
                    eps_pred_u = self.model(x_t[idx_u], t[idx_u], None)
                    loss_u = F.mse_loss(eps_pred_u, noise[idx_u])
                    loss = loss + loss_u * idx_u.numel()
                    denom += idx_u.numel()

                loss_diff = loss / max(denom, 1)

                phys_t_max = 100 
                idx_phys = idx_c[t[idx_c] < phys_t_max] if idx_c.numel() > 0 else idx_c

                if idx_phys.numel() > 0:
                    t_p = t[idx_phys]

                    sqrt_acp = extract(self.sqrt_alphas_cumprod, t, x_t.shape)[idx_phys]
                    sqrt_om = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)[idx_phys]

                    eps_pred_center = self.model(x_t[idx_phys], t_p, cond[idx_phys])
                    x0_pred = (x_t[idx_phys] - sqrt_om * eps_pred_center) / (sqrt_acp + 1e-8)

                    noise_prev = torch.randn_like(omega_prev[idx_phys])
                    noise_next = torch.randn_like(omega_next[idx_phys])

                    x_t_prev = self.q_sample(omega_prev[idx_phys], t_p, noise_prev)
                    x_t_next = self.q_sample(omega_next[idx_phys], t_p, noise_next)

                    eps_pred_prev = self.model(x_t_prev, t_p, cond_prev[idx_phys])
                    eps_pred_next = self.model(x_t_next, t_p, cond_next[idx_phys])

                    x_prev_pred = (x_t_prev - sqrt_om * eps_pred_prev) / (sqrt_acp + 1e-8)
                    x_next_pred = (x_t_next - sqrt_om * eps_pred_next) / (sqrt_acp + 1e-8)

                    scale = self.data_std + 1e-8
                    x0_phys = x0_pred * scale + self.data_mean
                    omega_prev_phys = x_prev_pred * scale + self.data_mean
                    omega_next_phys = x_next_pred * scale + self.data_mean

                    loss_smooth_space = self.spatial_smoothness_loss(x0_phys)
                    loss_smooth_time = self.temporal_second_difference_loss(
                        omega_prev_phys,
                        x0_phys,
                        omega_next_phys,
                    )

                    residual = self.pde_residual(omega_prev_phys, x0_phys, omega_next_phys)
                    loss_phys = F.mse_loss(residual, torch.zeros_like(residual))
                else:
                    loss_phys = torch.tensor(0.0, device=x_t.device)
                    loss_smooth_space = torch.tensor(0.0, device=x_t.device)
                    loss_smooth_time = torch.tensor(0.0, device=x_t.device)
                    residual = None

                lambda_phys_epoch = self.get_lambda_phys(
                    epoch=epoch,
                    total_epochs=self.cfg.epochs,
                    lambda_start=self.cfg.lambda_phys_start,
                    lambda_max=self.cfg.lambda_phys_max,
                    warmup_ratio=self.cfg.lambda_phys_warmup_ratio,
                )

                weighted_phys_raw = lambda_phys_epoch * loss_phys
                weighted_phys = torch.clamp(weighted_phys_raw, max=5.0)

                if loss_phys.item() > 1e8:
                    with torch.no_grad():
                        print("\n" + "=" * 80)
                        print(f"Large physics loss detected at epoch {epoch}, batch {batch_idx + 1}/{num_batches}")
                        print("=" * 80)

                        print(f"loss_diff:        {loss_diff.item():.6e}")
                        print(f"loss_phys:        {loss_phys.item():.6e}")
                        print(f"loss_phys weighted raw: {weighted_phys_raw.item():.6e}")
                        print(f"loss_phys weighted cap: {weighted_phys.item():.6e}")
                        print(f"loss_smooth_space:{loss_smooth_space.item():.6e}")
                        print(f"loss_smooth_time: {loss_smooth_time.item():.6e}")

                        print("\nResidual stats:")
                        print(f"  residual abs mean: {residual.abs().mean().item():.6e}")
                        print(f"  residual abs max:  {residual.abs().max().item():.6e}")

                        print("\nPrediction field stats:")
                        for name, w in [
                            ("omega_prev_phys", omega_prev_phys),
                            ("x0_phys", x0_phys),
                             ("omega_next_phys", omega_next_phys),
                        ]:
                            print(
                                f"  {name}: "
                                f"mean={w.mean().item():.6e}, "
                                f"std={w.std().item():.6e}, "
                                f"min={w.min().item():.6e}, "
                                f"max={w.max().item():.6e}, "
                                f"absmax={w.abs().max().item():.6e}"
                            )

                        print("\nTemporal jump stats:")
                        dt_prev = x0_phys - omega_prev_phys
                        dt_next = omega_next_phys - x0_phys
                        dtt = omega_next_phys - 2.0 * x0_phys + omega_prev_phys

                        print(f"  |x0 - prev| mean: {dt_prev.abs().mean().item():.6e}")
                        print(f"  |x0 - prev| max:  {dt_prev.abs().max().item():.6e}")
                        print(f"  |next - x0| mean: {dt_next.abs().mean().item():.6e}")
                        print(f"  |next - x0| max:  {dt_next.abs().max().item():.6e}")
                        print(f"  |dtt| mean:       {dtt.abs().mean().item():.6e}")
                        print(f"  |dtt| max:        {dtt.abs().max().item():.6e}")

                        print("=" * 80 + "\n")

                loss = (
                            loss_diff
                            + weighted_phys
                            + lambda_smooth_space * loss_smooth_space
                            + lambda_smooth_time * loss_smooth_time
                        )

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Skipping bad batch at batch {batch_idx + 1}")
                    print(f"  Diff: {loss_diff.item()} | Phys: {loss_phys.item()}")
                    continue

            self.scaler.scale(loss).backward()
            if self.cfg.grad_clip is not None:
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.scaler.step(self.opt)
            self.scaler.update()

            total_loss += float(loss.item()) * B
            total_diff_loss += float(loss_diff.item()) * B
            total_phys_loss += float(loss_phys.item()) * B
            total_smooth_space_loss += float(loss_smooth_space.item()) * B
            total_smooth_time_loss += float(loss_smooth_time.item()) * B
            n += B

            if (batch_idx + 1) % max(1, num_batches // 10) == 0 or (batch_idx + 1) % 10 == 0:
                current_avg_loss = total_loss / max(n, 1)
                current_avg_diff = total_diff_loss / max(n, 1)
                current_avg_phys = total_phys_loss / max(n, 1)
                print(
                        f"    Batch {batch_idx + 1}/{num_batches} | "
                        f"Loss: {loss.item():.6f} | "
                        f"Diff: {loss_diff.item():.6f} | "
                        f"WeightedPhys: {weighted_phys.item():.6f} | "
                        f"LambdaPhys: {lambda_phys_epoch:.2e} | "
                        f"Space: {loss_smooth_space.item():.6f} | "
                        f"TimeAccel: {loss_smooth_time.item():.6f} | "
                        #f"Avg Loss: {current_avg_loss:.6f} | "
                        f"Avg Diff: {current_avg_diff:.6f} | "
                        f"Avg Phys: {current_avg_phys:.6f}"
                    )

        avg_loss = total_loss / max(n, 1)
        avg_diff_loss = total_diff_loss / max(n, 1)
        avg_phys_loss = total_phys_loss / max(n, 1)
        avg_smooth_space_loss = total_smooth_space_loss / max(n, 1)
        avg_smooth_time_loss = total_smooth_time_loss / max(n, 1)
        print(
            f"  Epoch {epoch} complete | Average Loss: {avg_loss:.6f} | "
            f"Average Diff: {avg_diff_loss:.6f} | Average Phys: {avg_phys_loss:.6f} | "
            f"Spatial nonsmoothness: {avg_smooth_space_loss:.6f} | "
            f"Time acceleration: {avg_smooth_time_loss:.6f}"
        )
        return (
            avg_loss,
            avg_diff_loss,
            avg_phys_loss,
            avg_smooth_space_loss,
            avg_smooth_time_loss,
        )

    @torch.no_grad()
    def eval_recon_mse(self, loader: DataLoader, num_batches: int = 2) -> float:
        """
        Quick eval: sample reconstructions with CFG and compute MSE vs ground truth in normalized space.
        Note: DDPM sampling is slow; we only do a few batches.
        """
        self.model.eval()
        mses = []
        for i, (_, x0, _, _, cond, _) in enumerate(loader):
            if i >= num_batches:
                break
            x0 = x0.to(self.device)
            cond = cond.to(self.device)
            B = x0.size(0)
            x_hat = self.sample_cfg(cond=cond, guidance_scale=self.cfg.guidance_scale, shape=(B, 1, 64, 64))
            mse = F.mse_loss(x_hat, x0).item()
            mses.append(mse)
        return float(np.mean(mses)) if mses else float("nan")


# -------------------------
# Main: build loaders, train, test sample
# -------------------------
def run_training(
    data,  # np.ndarray or torch.Tensor of shape (N,64,64)
    out_dir: str = "/content/drive/MyDrive/Lab/CondDiff",
    seed: int = 0,
):
    os.makedirs(out_dir, exist_ok=True)
    seed_everything(seed)
    device = default_device()
    print("Device:", device)
    
    # Show checkpoint location clearly
    ckpt_path = os.path.join(out_dir, "best.pt")
    print(f"\n{'='*60}")
    print(f"CHECKPOINT LOCATION")
    print(f"{'='*60}")
    print(f"Checkpoint directory: {os.path.abspath(out_dir)}")
    print(f"Checkpoint file: {os.path.abspath(ckpt_path)}")
    print(f"{'='*60}\n")

    # Convert data to torch
    if isinstance(data, np.ndarray):
        data_t = torch.from_numpy(data)
    elif torch.is_tensor(data):
        data_t = data
    else:
        raise TypeError("data must be a numpy array or torch tensor")

    assert data_t.ndim == 3 and data_t.shape[1:] == (64, 64), f"Expected (N,64,64), got {data_t.shape}"
    N = data_t.shape[0]
    n_train = int(0.8 * N)

    train_full = data_t[:n_train]
    test_full = data_t[n_train:]

    # Compute normalization stats from TRAIN ONLY
    train_mean = train_full.float().mean().item()
    train_std = train_full.float().std().item()
    print(f"Train mean={train_mean:.6f}, std={train_std:.6f}")

    train_ds = NavierStokesSparseDataset(train_full, mean=train_mean, std=train_std, sensor_stride=8)
    test_ds = NavierStokesSparseDataset(test_full, mean=train_mean, std=train_std, sensor_stride=8)

    cfg = DiffusionConfig(
        T=1000,
        beta_schedule="cosine",
        drop_prob=0,
        lr=2e-4,
        batch_size=64,
        num_workers=0,  # Set to 0 for macOS compatibility (multiprocessing issues)
        grad_clip=1.0,
        epochs=30,
        guidance_scale=1.0,
        use_amp=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = ConditionalDDPM(T=cfg.T, emb_dim=256, base_ch=64)
    trainer = DDPMTrainer(model, cfg, device, data_mean=train_mean, data_std=train_std)

    ckpt_path = os.path.join(out_dir, "best.pt")
    
    print(f"\n{'='*60}")
    print(f"Starting Training")
    print(f"{'='*60}")
    print(f"Dataset: {N} samples ({n_train} train, {N - n_train} test)")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Epochs: {cfg.epochs}")
    print(f"Device: {device}")
    print(f"Checkpoint directory: {os.path.abspath(out_dir)}")
    print(f"Checkpoint will be saved to: {os.path.abspath(ckpt_path)}")
    print(f"{'='*60}\n")

    best_test = float("inf")

    # ---- MLflow setup ----
    mlflow.set_tracking_uri(f"file:{out_dir}/mlruns")  # simplest: store runs next to your checkpoints
    mlflow.set_experiment("pidm_2dns")

    with mlflow.start_run(run_name="pidm_sparse_conditional_ddpm"):
        run_id = mlflow.active_run().info.run_id
        # log hyperparameters
        mlflow.log_params(cfg.__dict__)
        mlflow.log_param("seed", seed)
        mlflow.log_param("train_mean", train_mean)
        mlflow.log_param("train_std", train_std)
        log_physics_mlflow_params(grid_size=64)

        best_test = float("inf")

        for epoch in range(1, cfg.epochs + 1):
            t0 = time.time()
            train_loss, train_loss_data, train_loss_physics, train_loss_spatial_nonsmoothness, train_loss_time_acceleration = (
                trainer.train_one_epoch(train_loader, epoch)
            )

            ckpt = {
                "model": trainer.model.state_dict(),
                "cfg": cfg.__dict__,
                "train_mean": train_mean,
                "train_std": train_std,
                "last_epoch": epoch,
                "optimizer": trainer.opt.state_dict(),
                "scaler": trainer.scaler.state_dict(),
                "best_test_recon_mse": best_test,
                "mlflow_run_id": run_id,
            }
            torch.save(ckpt, os.path.join(out_dir, "conditional.pt"))
            print("Saved conditional.pt (pre-eval).")

            test_mse = trainer.eval_recon_mse(test_loader, num_batches=2)
            dt = time.time() - t0

            mlflow.log_metric("train_loss", float(train_loss), step=epoch)
            mlflow.log_metric("train_loss_data", float(train_loss_data), step=epoch)
            mlflow.log_metric("train_loss_physics", float(train_loss_physics), step=epoch)
            mlflow.log_metric("train_loss_spatial_nonsmoothness", float(train_loss_spatial_nonsmoothness), step=epoch)
            mlflow.log_metric("train_loss_time_acceleration", float(train_loss_time_acceleration), step=epoch)
            mlflow.log_metric("test_recon_mse", float(test_mse), step=epoch)

            print(
                f"Epoch {epoch:03d} | train_loss={train_loss:.6f} "
                f"(data={train_loss_data:.6f}, phys={train_loss_physics:.6f}, "
                f"space={train_loss_spatial_nonsmoothness:.6f}, "
                f"time_accel={train_loss_time_acceleration:.6f}) | "
                f"test_recon_mse~={test_mse:.6f} | {dt:.1f}s"
            )

            if math.isnan(test_mse):
                print("WARNING: test_mse is NaN; saving anyway.")
                test_mse = float("inf")  # keep best logic sane

            # save best
            if epoch ==1 or test_mse < best_test:
                old_best = best_test
                best_test = test_mse
                ckpt = {
                    "model": trainer.model.state_dict(),
                    "cfg": cfg.__dict__,
                    "train_mean": train_mean,
                    "train_std": train_std,
                    "last_epoch": epoch,
                    "optimizer": trainer.opt.state_dict(),
                    "scaler": trainer.scaler.state_dict(),
                    "best_test_recon_mse": best_test,
                    "mlflow_run_id": run_id,
                }
                ckpt_path = os.path.join(out_dir, "best.pt")
                
                # Save checkpoint
                torch.save(ckpt, ckpt_path)
                mlflow.log_artifact(os.path.join(out_dir, "conditional.pt"), artifact_path="checkpoints")
                # when best:
                mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")
                
                # Verify checkpoint was saved
                if os.path.exists(ckpt_path):
                    file_size = os.path.getsize(ckpt_path) / (1024 * 1024)  # Size in MB
                    print(f"  ✓ NEW BEST! Checkpoint saved: {os.path.abspath(ckpt_path)} ({file_size:.2f} MB)")
                    print(f"    Test MSE improved: {old_best:.6f} → {best_test:.6f}")
                else:
                    print(f"  ✗ WARNING: Checkpoint file not found after saving!")
                

    print("Done. Best approx test recon MSE:", best_test)
    final_ckpt_path = os.path.join(out_dir, "best.pt")
    last_ckpt_path = os.path.join(out_dir, "conditional.pt")

    # Save final checkpoint (last epoch, regardless of whether it's best)
    torch.save(
        {
            "model": trainer.model.state_dict(),
            "cfg": cfg.__dict__,
            "train_mean": train_mean,
            "train_std": train_std,
            "last_epoch": cfg.epochs,
            "optimizer": trainer.opt.state_dict(),
            "scaler": trainer.scaler.state_dict(),
            "best_test_recon_mse": best_test,
            "mlflow_run_id": run_id,
        },
        last_ckpt_path,
    )
    
    # Verify both checkpoints
    if os.path.exists(final_ckpt_path):
        file_size = os.path.getsize(final_ckpt_path) / (1024 * 1024)
        print(f"Best checkpoint: {os.path.abspath(final_ckpt_path)} ({file_size:.2f} MB)")
    else:
        print(f"WARNING: Best checkpoint not found at {final_ckpt_path}")
    
    if os.path.exists(last_ckpt_path):
        file_size = os.path.getsize(last_ckpt_path) / (1024 * 1024)
        print(f"Last checkpoint: {os.path.abspath(last_ckpt_path)} ({file_size:.2f} MB)")
    else:
        print(f"WARNING: Last checkpoint not found at {last_ckpt_path}")
    
    return final_ckpt_path, (train_mean, train_std), cfg


def _diffusion_config_from_ckpt(cfg_dict: Dict[str, Any]) -> DiffusionConfig:
    fields = set(DiffusionConfig.__dataclass_fields__.keys())
    return DiffusionConfig(**{k: v for k, v in cfg_dict.items() if k in fields})


def mlflow_last_logged_step(tracking_uri: str, run_id: str, metric_name: str = "train_loss") -> int:
    """Largest `step` seen for `metric_name` in the given run (0 if none)."""
    client = MlflowClient(tracking_uri)
    history = client.get_metric_history(run_id, metric_name)
    if not history:
        return 0
    return max(h.step for h in history)


def run_training_resume(
    data,
    ckpt_path: str,
    out_dir: str,
    additional_epochs: int,
    mlflow_run_id: Optional[str] = None,
    seed: int = 0,
    epoch_offset: Optional[int] = None,
):
    """
    Resume training from a checkpoint produced by `run_training` (ddpm_sparse_cfg).

    - Reloads model weights and (if present) optimizer + GradScaler state.
    - Uses `train_mean` / `train_std` from the checkpoint (same normalization as original run).
    - If `mlflow_run_id` is set, continues that MLflow run and logs metrics with `step` after the
      last logged `train_loss` step (or `epoch_offset` if you pass it explicitly).
    - If `mlflow_run_id` is None, starts a new MLflow run named `cfg_conditional_ddpm_resume`.

    Returns:
        (best_ckpt_path, (train_mean, train_std), cfg)
    """
    os.makedirs(out_dir, exist_ok=True)
    seed_everything(seed)
    device = default_device()
    print("Device:", device)
    print(f"Resuming from checkpoint: {os.path.abspath(ckpt_path)}")

    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    train_mean = float(ckpt["train_mean"])
    train_std = float(ckpt["train_std"])
    cfg = _diffusion_config_from_ckpt(ckpt["cfg"])

    if isinstance(data, np.ndarray):
        data_t = torch.from_numpy(data)
    elif torch.is_tensor(data):
        data_t = data
    else:
        raise TypeError("data must be a numpy array or torch tensor")

    assert data_t.ndim == 3 and data_t.shape[1:] == (64, 64), f"Expected (N,64,64), got {data_t.shape}"
    N = data_t.shape[0]
    n_train = int(0.8 * N)
    train_full = data_t[:n_train]
    test_full = data_t[n_train:]

    train_ds = NavierStokesSparseDataset(train_full, mean=train_mean, std=train_std, sensor_stride=8)
    test_ds = NavierStokesSparseDataset(test_full, mean=train_mean, std=train_std, sensor_stride=8)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = ConditionalDDPM(T=cfg.T, emb_dim=256, base_ch=64)
    model.load_state_dict(ckpt["model"])
    trainer = DDPMTrainer(model, cfg, device, data_mean=train_mean, data_std=train_std)

    if "optimizer" in ckpt and ckpt["optimizer"] is not None:
        trainer.opt.load_state_dict(ckpt["optimizer"])
        print("Loaded optimizer state from checkpoint.")
    else:
        print("No optimizer state in checkpoint; optimizer reinitialized.")

    if "scaler" in ckpt and ckpt["scaler"] is not None:
        try:
            trainer.scaler.load_state_dict(ckpt["scaler"])
            print("Loaded GradScaler state from checkpoint.")
        except Exception as e:
            print(f"Could not load GradScaler state: {e}")

    # Must match run_training: file:{out_dir}/mlruns (use same absolute path for metric lookup)
    mlflow_uri = f"file:{os.path.abspath(os.path.join(out_dir, 'mlruns'))}"
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("pidm_2dns")

    if mlflow_run_id:
        if epoch_offset is not None:
            start_step = int(epoch_offset)
        else:
            start_step = mlflow_last_logged_step(mlflow_uri, mlflow_run_id, "train_loss")
            if start_step == 0:
                start_step = int(ckpt.get("last_epoch", 0))
        print(f"MLflow run {mlflow_run_id}: logging new epochs at step > {start_step} (next step = {start_step + 1})")
    else:
        start_step = int(ckpt.get("last_epoch", 0))
        print(f"Starting new MLflow run; metric steps start after {start_step} (next step = {start_step + 1}).")

    ckpt_best_path = os.path.join(out_dir, "best.pt")
    best_test = float(ckpt.get("best_test_recon_mse", float("inf")))

    run_ctx = (
        mlflow.start_run(run_id=mlflow_run_id)
        if mlflow_run_id
        else mlflow.start_run(run_name="pidm_sparse_conditional_ddpm_resume")
    )

    with run_ctx:
        active_run_id = mlflow.active_run().info.run_id
        if not mlflow_run_id:
            mlflow.log_param("resumed_from_ckpt", os.path.abspath(ckpt_path))
            mlflow.log_param("seed", seed)
            mlflow.log_param("train_mean", train_mean)
            mlflow.log_param("train_std", train_std)
            mlflow.log_param("additional_epochs", additional_epochs)
            log_physics_mlflow_params(grid_size=64)

        for i in range(1, additional_epochs + 1):
            log_step = start_step + i
            t0 = time.time()
            train_loss, train_loss_data, train_loss_physics, train_loss_spatial_nonsmoothness, train_loss_time_acceleration = (
                trainer.train_one_epoch(train_loader, i)
            )

            payload = {
                "model": trainer.model.state_dict(),
                "cfg": cfg.__dict__,
                "train_mean": train_mean,
                "train_std": train_std,
                "last_epoch": log_step,
                "optimizer": trainer.opt.state_dict(),
                "scaler": trainer.scaler.state_dict() if hasattr(trainer.scaler, "state_dict") else None,
                "best_test_recon_mse": best_test,
                "mlflow_run_id": active_run_id,
            }
            torch.save(payload, os.path.join(out_dir, "conditional.pt"))

            test_mse = trainer.eval_recon_mse(test_loader, num_batches=2)
            dt = time.time() - t0

            mlflow.log_metric("train_loss", float(train_loss), step=log_step)
            mlflow.log_metric("train_loss_data", float(train_loss_data), step=log_step)
            mlflow.log_metric("train_loss_physics", float(train_loss_physics), step=log_step)
            mlflow.log_metric("train_loss_spatial_nonsmoothness", float(train_loss_spatial_nonsmoothness), step=log_step)
            mlflow.log_metric("train_loss_time_acceleration", float(train_loss_time_acceleration), step=log_step)
            mlflow.log_metric("test_recon_mse", float(test_mse), step=log_step)

            print(
                f"Epoch (resume {i}/{additional_epochs}) global_step={log_step} | "
                f"train_loss={train_loss:.6f} (data={train_loss_data:.6f}, phys={train_loss_physics:.6f}, "
                f"space={train_loss_spatial_nonsmoothness:.6f}, "
                f"time_accel={train_loss_time_acceleration:.6f}) | "
                f"test_recon_mse~={test_mse:.6f} | {dt:.1f}s"
            )

            if math.isnan(test_mse):
                test_mse = float("inf")

            if test_mse < best_test:
                best_test = test_mse
                payload["best_test_recon_mse"] = best_test
                torch.save(payload, ckpt_best_path)
                mlflow.log_artifact(os.path.join(out_dir, "conditional.pt"), artifact_path="checkpoints")
                mlflow.log_artifact(ckpt_best_path, artifact_path="checkpoints")
                print(f"  ✓ NEW BEST: {os.path.abspath(ckpt_best_path)}")

    torch.save(
        {
            "model": trainer.model.state_dict(),
            "cfg": cfg.__dict__,
            "train_mean": train_mean,
            "train_std": train_std,
            "last_epoch": start_step + additional_epochs,
            "optimizer": trainer.opt.state_dict(),
            "scaler": trainer.scaler.state_dict() if hasattr(trainer.scaler, "state_dict") else None,
            "best_test_recon_mse": best_test,
            "mlflow_run_id": active_run_id,
        },
        os.path.join(out_dir, "conditional.pt"),
    )

    return ckpt_best_path, (train_mean, train_std), cfg


# -------------------------
# Example usage:
# After importing your data into variable `data` with shape (100000,64,64)
# -------------------------
# ckpt_path, (mean, std), cfg = run_training(data)


# -------------------------
# Loading + sampling example (reconstruct full field from 64x64 y on test)
# -------------------------
@torch.no_grad()
def load_and_sample(
    ckpt_path: str,
    data,
    num_samples: int = 8,
    guidance_scale: float = 1.0,
    sensor_stride: int = 8,
):
    device = default_device()

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_dict = ckpt["cfg"]
    T = cfg_dict["T"]

    model = ConditionalDDPM(T=T, emb_dim=256, base_ch=64).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    cfg = _diffusion_config_from_ckpt(cfg_dict)
    trainer = DDPMTrainer(
        model, cfg, device,
        data_mean=float(ckpt["train_mean"]),
        data_std=float(ckpt["train_std"]),
    )

    if isinstance(data, np.ndarray):
        data_t = torch.from_numpy(data)
    else:
        data_t = data

    N = data_t.shape[0]
    n_train = int(0.8 * N)
    test_full = data_t[n_train:].float()

    mean = ckpt["train_mean"]
    std = ckpt["train_std"]

    # pick center indices k with neighbors available in test split
    max_k = test_full.shape[0] - 2
    k_idx = torch.randint(1, max_k + 1, (num_samples,))
    x0 = test_full[k_idx]                        # (B,64,64)
    x0n = (x0 - mean) / (std + 1e-8)          # normalized

    mask = torch.zeros_like(x0n)
    mask[:, ::sensor_stride, ::sensor_stride] = 1.0   # 8x8 lattice if stride=8

    y_sparse = x0n * mask
    cond = torch.stack([y_sparse, mask], dim=1)       # (B,2,64,64)

    xhat = trainer.sample_cfg(cond=cond.to(device),
                              guidance_scale=guidance_scale,
                              shape=(num_samples, 1, 64, 64))

    xhat = xhat.squeeze(1) * (std + 1e-8) + mean

    # optional: return sparse observation on original scale for plotting
    y_sparse_denorm = y_sparse * (std + 1e-8) + mean * mask

    return x0.cpu(), xhat.cpu(), y_sparse_denorm.cpu(), mask.cpu()


# Example:
# x_true, x_pred, y_sparse = load_and_sample("./ddpm_sparse_cfg/best.pt", data, num_samples=4, guidance_scale=4.0)
# print(x_true.shape, x_pred.shape, y_sparse.shape)