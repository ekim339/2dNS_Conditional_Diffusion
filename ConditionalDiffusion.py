# ============================================
# Conditional DDPM (Sparse 8x8 -> Full 64x64)
# Classifier-Free Guidance (CFG) in PyTorch
#
# Assumptions:
# - You already have `data` as a NumPy array or torch Tensor of shape (N, 64, 64),
#   e.g. N=100000, dtype float32 (or convertible).
# - Data are single-channel scalar fields (e.g., vorticity).
#
# What this does:
# - Splits first 80% train, last 20% test (as requested).
# - Creates sparse observations y in R^{8x8} by uniform subsampling (fixed grid).
# - Trains a conditional DDPM with CFG: randomly drops condition during training.
# - Provides sampling with CFG to reconstruct 64x64 fields from 8x8 observations.
#
# Notes:
# - This "directly encodes" the 8x8 into the network via a small conv encoder
#   that produces an embedding used to FiLM-modulate U-Net ResBlocks.
# - This does NOT enforce exact sensor consistency during sampling (pure CFG).
#   (You can add a projection step if needed.)
# ============================================

import math
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


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


# -------------------------
# Dataset: fixed split + fixed uniform sampling grid (8x8)
# -------------------------
class NavierStokesSparseDataset(Dataset):
    def __init__(
        self,
        full_fields: torch.Tensor,  # (N, 64, 64)
        mean: float,
        std: float,
        sensor_stride: int = 8,      # 64/8 = 8
    ):
        """
        We assume uniform sensor grid (8x8) over 64x64:
          take pixels at [0, 8, 16, ..., 56] in each axis (stride=8).
        """
        assert full_fields.ndim == 3 and full_fields.shape[1] == 64 and full_fields.shape[2] == 64
        self.x = full_fields.float()
        self.mean = float(mean)
        self.std = float(std)
        self.sensor_stride = int(sensor_stride)

        # Indices for uniform grid
        coords = torch.arange(0, 64, self.sensor_stride)
        assert len(coords) == 8, "Expected 8 points per axis for 8x8 sensors."
        self.registered_coords = coords

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        x0 = self.x[idx]  # (64, 64)
        # normalize
        x0 = (x0 - self.mean) / (self.std + 1e-8)

        # get sparse observation y (8, 8) by uniform subsampling
        c = self.registered_coords
        y = x0[c][:, c]  # (8, 8)

        return x0.unsqueeze(0), y.unsqueeze(0)  # (1,64,64), (1,8,8)


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
# Condition encoder: directly encode (1,8,8) -> embedding vector
# -------------------------
class CondEncoder8x8(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        # Very small conv encoder to preserve spatial structure
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2),  # 8->4
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, padding=1, stride=2), # 4->2
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(128, emb_dim)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        y: (B,1,8,8)
        returns: (B, emb_dim)
        """
        h = self.net(y).flatten(1)
        return self.proj(h)


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
        self.in_conv = nn.Conv2d(1, base_ch, 3, padding=1)

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
        # x: (B,1,64,64)
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

        self.cond_enc = CondEncoder8x8(emb_dim)
        self.null_cond = nn.Parameter(torch.zeros(emb_dim))  # learned unconditional embedding

        self.unet = UNet64FiLM(base_ch=base_ch, emb_dim=emb_dim)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor]) -> torch.Tensor:
        """
        x_t: (B,1,64,64)
        t:   (B,) int64
        y:   (B,1,8,8) or None for unconditional
        """
        et = self.time_mlp(self.time_emb(t))  # (B, emb_dim)

        if y is None:
            ey = self.null_cond[None, :].expand(x_t.size(0), -1)
        else:
            ey = self.cond_enc(y)

        emb = et + ey
        return self.unet(x_t, emb)


# -------------------------
# DDPM Diffusion wrapper (training + sampling)
# -------------------------
@dataclass
class DiffusionConfig:
    T: int = 1000
    beta_schedule: str = "cosine"
    drop_prob: float = 0.1     # CFG condition dropout probability
    lr: float = 2e-4
    batch_size: int = 64
    num_workers: int = 0  # Set to 0 for macOS compatibility (multiprocessing issues)
    grad_clip: float = 1.0
    epochs: int = 10
    guidance_scale: float = 10.0  # CFG sampling scale
    use_amp: bool = True


class DDPMTrainer:
    def __init__(self, model: ConditionalDDPM, cfg: DiffusionConfig, device: torch.device):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device

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
    def sample_cfg(self, y: torch.Tensor, guidance_scale: float, shape: Tuple[int, int, int, int]) -> torch.Tensor:
        """
        CFG sampling:
          eps = eps_uncond + s*(eps_cond - eps_uncond)
        y: (B,1,8,8)
        shape: (B,1,64,64)
        returns x0 samples (B,1,64,64)
        """
        self.model.eval()
        x = torch.randn(shape, device=self.device)

        for i in reversed(range(self.cfg.T)):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)

            # unconditional and conditional eps
            eps_u = self.model(x, t, None)
            eps_c = self.model(x, t, y)
            eps = eps_u + guidance_scale * (eps_c - eps_u)

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

    def train_one_epoch(self, loader: DataLoader, epoch: int):
        self.model.train()
        total_loss = 0.0
        n = 0
        num_batches = len(loader)
        
        print(f"  Starting epoch {epoch} ({num_batches} batches)...")

        for batch_idx, (x0, y) in enumerate(loader):
            x0 = x0.to(self.device)  # (B,1,64,64)
            y = y.to(self.device)    # (B,1,8,8)

            B = x0.size(0)
            t = torch.randint(0, self.cfg.T, (B,), device=self.device, dtype=torch.long)
            noise = torch.randn_like(x0)
            x_t = self.q_sample(x0, t, noise)

            # CFG dropout
            cond_mask = (torch.rand(B, device=self.device) > self.cfg.drop_prob)
            # If dropped, pass y=None for those samples.
            # We'll do it in two batches for simplicity and correctness.

            self.opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(self.cfg.use_amp and self.device.type == "cuda")):
                # conditional subset
                idx_c = torch.nonzero(cond_mask, as_tuple=False).squeeze(1)
                idx_u = torch.nonzero(~cond_mask, as_tuple=False).squeeze(1)

                loss = 0.0
                denom = 0

                if idx_c.numel() > 0:
                    eps_pred_c = self.model(x_t[idx_c], t[idx_c], y[idx_c])
                    loss_c = F.mse_loss(eps_pred_c, noise[idx_c])
                    loss = loss + loss_c * idx_c.numel()
                    denom += idx_c.numel()

                if idx_u.numel() > 0:
                    eps_pred_u = self.model(x_t[idx_u], t[idx_u], None)
                    loss_u = F.mse_loss(eps_pred_u, noise[idx_u])
                    loss = loss + loss_u * idx_u.numel()
                    denom += idx_u.numel()

                loss = loss / max(denom, 1)

            self.scaler.scale(loss).backward()
            if self.cfg.grad_clip is not None:
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.scaler.step(self.opt)
            self.scaler.update()

            total_loss += float(loss.item()) * B
            n += B
            
            # Print progress every 10% of batches or every 10 batches, whichever is more frequent
            if (batch_idx + 1) % max(1, num_batches // 10) == 0 or (batch_idx + 1) % 10 == 0:
                current_avg_loss = total_loss / max(n, 1)
                print(f"    Batch {batch_idx + 1}/{num_batches} | Loss: {loss.item():.6f} | Avg Loss: {current_avg_loss:.6f}")

        avg_loss = total_loss / max(n, 1)
        print(f"  Epoch {epoch} complete | Average Loss: {avg_loss:.6f}")
        return avg_loss

    @torch.no_grad()
    def eval_recon_mse(self, loader: DataLoader, num_batches: int = 2) -> float:
        """
        Quick eval: sample reconstructions with CFG and compute MSE vs ground truth in normalized space.
        Note: DDPM sampling is slow; we only do a few batches.
        """
        self.model.eval()
        mses = []
        for i, (x0, y) in enumerate(loader):
            if i >= num_batches:
                break
            x0 = x0.to(self.device)
            y = y.to(self.device)
            B = x0.size(0)
            x_hat = self.sample_cfg(y=y, guidance_scale=self.cfg.guidance_scale, shape=(B, 1, 64, 64))
            mse = F.mse_loss(x_hat, x0).item()
            mses.append(mse)
        return float(np.mean(mses)) if mses else float("nan")


# -------------------------
# Main: build loaders, train, test sample
# -------------------------
def run_training(
    data,  # np.ndarray or torch.Tensor of shape (N,64,64)
    out_dir: str = "/Users/eugenekim/2dNS_Conditional_Diffusion/checkpoint",
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
        drop_prob=0.1,
        lr=2e-4,
        batch_size=64,
        num_workers=0,  # Set to 0 for macOS compatibility (multiprocessing issues)
        grad_clip=1.0,
        epochs=10,
        guidance_scale=4.0,
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
    trainer = DDPMTrainer(model, cfg, device)

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

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_loss = trainer.train_one_epoch(train_loader, epoch)
        
        ckpt = {
            "model": trainer.model.state_dict(),
            "cfg": cfg.__dict__,
            "train_mean": train_mean,
            "train_std": train_std,
        }
        torch.save(ckpt, os.path.join(out_dir, "conditional.pt"))
        print("Saved conditional.pt (pre-eval).")

        test_mse = trainer.eval_recon_mse(test_loader, num_batches=2)
        dt = time.time() - t0

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | test_recon_mse~={test_mse:.6f} | {dt:.1f}s")

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
            }
            ckpt_path = os.path.join(out_dir, "best.pt")
            
            # Save checkpoint
            torch.save(ckpt, ckpt_path)
            
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


# -------------------------
# Example usage:
# After importing your data into variable `data` with shape (100000,64,64)
# -------------------------
# ckpt_path, (mean, std), cfg = run_training(data)


# -------------------------
# Loading + sampling example (reconstruct full field from sparse y on test)
# -------------------------
@torch.no_grad()
def load_and_sample(
    ckpt_path: str,
    data,  # same shape (N,64,64) to pull test examples from
    num_samples: int = 8,
    guidance_scale: float = 4.0,
):
    device = default_device()

    # load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_dict = ckpt["cfg"]
    T = cfg_dict["T"]

    model = ConditionalDDPM(T=T, emb_dim=256, base_ch=64).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # re-create diffusion wrapper
    cfg = DiffusionConfig(**cfg_dict)
    trainer = DDPMTrainer(model, cfg, device)

    # prepare test split as requested: last 20%
    if isinstance(data, np.ndarray):
        data_t = torch.from_numpy(data)
    else:
        data_t = data
    N = data_t.shape[0]
    n_train = int(0.8 * N)
    test_full = data_t[n_train:].float()

    mean = ckpt["train_mean"]
    std = ckpt["train_std"]

    # pick some random test examples
    idx = torch.randint(0, test_full.shape[0], (num_samples,))
    x0 = test_full[idx]  # (B,64,64)
    x0n = (x0 - mean) / (std + 1e-8)

    # build y (8,8)
    coords = torch.arange(0, 64, 8)
    y = x0n[:, coords][:, :, coords]  # (B,8,8)

    x0n = x0n.unsqueeze(1).to(device)  # (B,1,64,64)
    y = y.unsqueeze(1).to(device)      # (B,1,8,8)

    xhat = trainer.sample_cfg(y=y, guidance_scale=guidance_scale, shape=(num_samples, 1, 64, 64))
    # unnormalize to original scale
    xhat = xhat.squeeze(1) * (std + 1e-8) + mean  # (B,64,64)
    x0 = x0  # original

    return x0.cpu(), xhat.cpu(), y.squeeze(1).cpu()  # y is normalized 8x8


# Example:
# x_true, x_pred, y_sparse = load_and_sample("./ddpm_sparse_cfg/best.pt", data, num_samples=4, guidance_scale=4.0)
# print(x_true.shape, x_pred.shape, y_sparse.shape)