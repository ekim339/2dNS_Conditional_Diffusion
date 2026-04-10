# ============================================
# Conditional DDPM — 3 consecutive frames as a 3D tensor
#
# Sparse condition: (B, 1, 3, 8, 8) — three consecutive 8×8 sensor grids
# Full target:      (B, 1, 3, 64, 64) — three consecutive 64×64 fields
#
# Diffusion runs on 5D tensors; U-Net uses Conv3d with spatial-only pooling
# (stride (1,2,2)) so temporal depth stays 3 throughout.
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
import mlflow


NUM_TEMPORAL = 3


def seed_everything(seed: int = 0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_beta_schedule(T: int, kind: str = "cosine") -> torch.Tensor:
    if kind == "linear":
        return torch.linspace(1e-4, 2e-2, T, dtype=torch.float32)
    if kind != "cosine":
        raise ValueError(f"Unknown schedule: {kind}")
    s = 0.008
    steps = T + 1
    x = torch.linspace(0, T, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 1e-8, 0.999).float()


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    b = t.shape[0]
    out = a.gather(0, t).reshape(b, *((1,) * (len(x_shape) - 1)))
    return out


# -------------------------
# Dataset: 3 consecutive (64,64) frames → (1,3,64,64) full + (1,3,8,8) sparse
# -------------------------
class NavierStokesSparse3FrameDataset(Dataset):
    def __init__(
        self,
        full_fields: torch.Tensor,
        mean: float,
        std: float,
        sensor_stride: int = 8,
    ):
        assert full_fields.ndim == 3 and full_fields.shape[1] == 64 and full_fields.shape[2] == 64
        self.x = full_fields.float()
        self.mean = float(mean)
        self.std = float(std)
        self.sensor_stride = int(sensor_stride)
        coords = torch.arange(0, 64, self.sensor_stride)
        assert len(coords) == 8, "Expected 8 points per axis for 8x8 sensors."
        self.registered_coords = coords

    def __len__(self):
        return max(0, self.x.shape[0] - (NUM_TEMPORAL - 1))

    def __getitem__(self, idx: int):
        c = self.registered_coords
        frames_full = []
        frames_sparse = []
        for j in range(NUM_TEMPORAL):
            xj = self.x[idx + j]
            xj = (xj - self.mean) / (self.std + 1e-8)
            yj = xj[c][:, c]
            frames_full.append(xj)
            frames_sparse.append(yj)
        # (3,H,W) -> (1,1,3,H,W) for Conv3d (B,C,D,H,W)
        x0 = torch.stack(frames_full, dim=0).unsqueeze(0).unsqueeze(0)
        y = torch.stack(frames_sparse, dim=0).unsqueeze(0).unsqueeze(0)
        return x0, y


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
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
# Condition encoder: (B, 1, 3, 8, 8) → embedding
# -------------------------
class CondEncoder3x8x8(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv3d(32, 64, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.SiLU(),
            nn.Conv3d(64, 128, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.SiLU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.proj = nn.Linear(128, emb_dim)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        y: (B, 1, 3, 8, 8)
        """
        h = self.net(y).flatten(1)
        return self.proj(h)


class ResBlockFiLM3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, groups: int = 8):
        super().__init__()
        g1 = min(groups, in_ch)
        while g1 > 1 and in_ch % g1 != 0:
            g1 -= 1
        g2 = min(groups, out_ch)
        while g2 > 1 and out_ch % g2 != 0:
            g2 -= 1
        self.norm1 = nn.GroupNorm(g1, in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(g2, out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, 2 * out_ch)
        self.act = nn.SiLU()
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        gamma_beta = self.emb_proj(emb)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma[:, :, None, None, None]
        beta = beta[:, :, None, None, None]
        h = self.norm2(h)
        h = h * (1 + gamma) + beta
        h = self.conv2(self.act(h))
        return h + self.skip(x)


class UNet3D64FiLM(nn.Module):
    """
    5D volume (B, 1, 3, 64, 64). Spatial down/up only: kernel (1,4,4), stride (1,2,2).
    """

    def __init__(self, base_ch: int = 64, emb_dim: int = 256):
        super().__init__()
        self.in_conv = nn.Conv3d(1, base_ch, 3, padding=1)

        self.rb1 = ResBlockFiLM3D(base_ch, base_ch, emb_dim)
        self.down1 = nn.Conv3d(base_ch, base_ch, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))

        self.rb2 = ResBlockFiLM3D(base_ch, base_ch * 2, emb_dim)
        self.down2 = nn.Conv3d(base_ch * 2, base_ch * 2, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))

        self.rb3 = ResBlockFiLM3D(base_ch * 2, base_ch * 4, emb_dim)
        self.down3 = nn.Conv3d(base_ch * 4, base_ch * 4, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))

        self.rb_mid1 = ResBlockFiLM3D(base_ch * 4, base_ch * 4, emb_dim)
        self.rb_mid2 = ResBlockFiLM3D(base_ch * 4, base_ch * 4, emb_dim)

        self.up3 = nn.ConvTranspose3d(base_ch * 4, base_ch * 4, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))
        self.rb_up3 = ResBlockFiLM3D(base_ch * 8, base_ch * 2, emb_dim)

        self.up2 = nn.ConvTranspose3d(base_ch * 2, base_ch * 2, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))
        self.rb_up2 = ResBlockFiLM3D(base_ch * 4, base_ch, emb_dim)

        self.up1 = nn.ConvTranspose3d(base_ch, base_ch, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))
        self.rb_up1 = ResBlockFiLM3D(base_ch * 2, base_ch, emb_dim)

        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_conv = nn.Conv3d(base_ch, 1, 3, padding=1)
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
        u3 = self.up3(mid)
        u3 = torch.cat([u3, h3], dim=1)
        u3 = self.rb_up3(u3, emb)
        u2 = self.up2(u3)
        u2 = torch.cat([u2, h2], dim=1)
        u2 = self.rb_up2(u2, emb)
        u1 = self.up1(u2)
        u1 = torch.cat([u1, h1], dim=1)
        u1 = self.rb_up1(u1, emb)
        return self.out_conv(self.act(self.out_norm(u1)))


class ConditionalDDPM3D(nn.Module):
    def __init__(self, T: int = 1000, emb_dim: int = 256, base_ch: int = 64):
        super().__init__()
        self.T = T
        self.time_emb = SinusoidalTimeEmbedding(emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.SiLU(),
            nn.Linear(emb_dim * 4, emb_dim),
        )
        self.cond_enc = CondEncoder3x8x8(emb_dim)
        self.null_cond = nn.Parameter(torch.zeros(emb_dim))
        self.unet = UNet3D64FiLM(base_ch=base_ch, emb_dim=emb_dim)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor]) -> torch.Tensor:
        """
        x_t: (B, 1, 3, 64, 64)
        t:   (B,)
        y:   (B, 1, 3, 8, 8) or None
        """
        et = self.time_mlp(self.time_emb(t))
        if y is None:
            ey = self.null_cond[None, :].expand(x_t.size(0), -1)
        else:
            ey = self.cond_enc(y)
        emb = et + ey
        return self.unet(x_t, emb)


@dataclass
class DiffusionConfig:
    T: int = 1000
    beta_schedule: str = "cosine"
    drop_prob: float = 0.1
    lr: float = 2e-4
    batch_size: int = 64
    num_workers: int = 0
    grad_clip: float = 1.0
    epochs: int = 10
    guidance_scale: float = 10.0
    use_amp: bool = True
    num_temporal: int = NUM_TEMPORAL


class DDPMTrainer:
    def __init__(self, model: ConditionalDDPM3D, cfg: DiffusionConfig, device: torch.device):
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

        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), alphas_cumprod[:-1]], dim=0)
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and device.type == "cuda"))

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return extract(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0 + \
               extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape) * noise

    def p_mean_variance(
        self, x_t: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        eps = self.model(x_t, t, y)
        sqrt_acp = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_om = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        x0_pred = (x_t - sqrt_om * eps) / (sqrt_acp + 1e-8)
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
    def sample_cfg(
        self,
        y: torch.Tensor,
        guidance_scale: float,
        shape: Tuple[int, int, int, int, int],
    ) -> torch.Tensor:
        """
        y: (B, 1, 3, 8, 8)
        shape: (B, 1, 3, 64, 64)
        """
        self.model.eval()
        x = torch.randn(shape, device=self.device)
        for i in reversed(range(self.cfg.T)):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            eps_u = self.model(x, t, None)
            eps_c = self.model(x, t, y)
            eps = eps_u + guidance_scale * (eps_c - eps_u)
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
                x = mean + torch.sqrt(var + 1e-8) * torch.randn_like(x)
            else:
                x = mean
        return x

    def train_one_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        n = 0
        num_batches = len(loader)
        print(f"  Starting epoch {epoch} ({num_batches} batches)...")

        for batch_idx, (x0, y) in enumerate(loader):
            x0 = x0.to(self.device)
            y = y.to(self.device)
            B = x0.size(0)
            t = torch.randint(0, self.cfg.T, (B,), device=self.device, dtype=torch.long)
            noise = torch.randn_like(x0)
            x_t = self.q_sample(x0, t, noise)
            cond_mask = torch.rand(B, device=self.device) > self.cfg.drop_prob
            self.opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(self.cfg.use_amp and self.device.type == "cuda")):
                idx_c = torch.nonzero(cond_mask, as_tuple=False).squeeze(1)
                idx_u = torch.nonzero(~cond_mask, as_tuple=False).squeeze(1)
                loss = 0.0
                denom = 0
                if idx_c.numel() > 0:
                    eps_c = self.model(x_t[idx_c], t[idx_c], y[idx_c])
                    loss = loss + F.mse_loss(eps_c, noise[idx_c]) * idx_c.numel()
                    denom += idx_c.numel()
                if idx_u.numel() > 0:
                    eps_u = self.model(x_t[idx_u], t[idx_u], None)
                    loss = loss + F.mse_loss(eps_u, noise[idx_u]) * idx_u.numel()
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
            if (batch_idx + 1) % max(1, num_batches // 10) == 0 or (batch_idx + 1) % 10 == 0:
                print(
                    f"    Batch {batch_idx + 1}/{num_batches} | Loss: {loss.item():.6f} | "
                    f"Avg: {total_loss / max(n, 1):.6f}"
                )

        avg_loss = total_loss / max(n, 1)
        print(f"  Epoch {epoch} complete | Average Loss: {avg_loss:.6f}")
        return avg_loss

    @torch.no_grad()
    def eval_recon_mse(self, loader: DataLoader, num_batches: int = 2) -> float:
        self.model.eval()
        mses = []
        Bt = self.cfg.num_temporal
        for i, (x0, y) in enumerate(loader):
            if i >= num_batches:
                break
            x0 = x0.to(self.device)
            y = y.to(self.device)
            B = x0.size(0)
            x_hat = self.sample_cfg(
                y=y,
                guidance_scale=self.cfg.guidance_scale,
                shape=(B, 1, Bt, 64, 64),
            )
            mses.append(F.mse_loss(x_hat, x0).item())
        return float(np.mean(mses)) if mses else float("nan")


def run_training(
    data,
    out_dir: str = "/content/drive/MyDrive/Lab/CondDiff",
    seed: int = 0,
):
    os.makedirs(out_dir, exist_ok=True)
    seed_everything(seed)
    device = default_device()
    print("Device:", device)

    ckpt_path = os.path.join(out_dir, "best.pt")
    print(f"\n{'='*60}\nCHECKPOINT LOCATION\n{'='*60}")
    print(f"Checkpoint directory: {os.path.abspath(out_dir)}")
    print(f"Checkpoint file: {os.path.abspath(ckpt_path)}\n{'='*60}\n")

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

    train_mean = train_full.float().mean().item()
    train_std = train_full.float().std().item()
    print(f"Train mean={train_mean:.6f}, std={train_std:.6f}")
    print(f"3D setup: condition (1,3,8,8), target (1,3,64,64) per sample")

    train_ds = NavierStokesSparse3FrameDataset(train_full, mean=train_mean, std=train_std, sensor_stride=8)
    test_ds = NavierStokesSparse3FrameDataset(test_full, mean=train_mean, std=train_std, sensor_stride=8)

    cfg = DiffusionConfig(
        T=1000,
        beta_schedule="cosine",
        drop_prob=0.1,
        lr=2e-4,
        batch_size=64,
        num_workers=0,
        grad_clip=1.0,
        epochs=10,
        guidance_scale=4.0,
        use_amp=True,
        num_temporal=NUM_TEMPORAL,
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

    model = ConditionalDDPM3D(T=cfg.T, emb_dim=256, base_ch=64)
    trainer = DDPMTrainer(model, cfg, device)

    print(f"\n{'='*60}\nStarting Training (3D tensor DDPM)\n{'='*60}")
    print(f"Dataset: {N} frames → {len(train_ds)} train triplets, {len(test_ds)} test triplets")
    print(f"Batch size: {cfg.batch_size}, Epochs: {cfg.epochs}\n{'='*60}\n")

    best_test = float("inf")
    mlflow.set_tracking_uri(f"file:{out_dir}/mlruns")
    mlflow.set_experiment("conditional_ddpm_2dns_3d")

    with mlflow.start_run(run_name="cfg_conditional_ddpm_3d"):
        mlflow.log_params(cfg.__dict__)
        mlflow.log_param("seed", seed)
        mlflow.log_param("train_mean", train_mean)
        mlflow.log_param("train_std", train_std)

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
            mlflow.log_metric("train_loss", float(train_loss), step=epoch)
            mlflow.log_metric("test_recon_mse", float(test_mse), step=epoch)
            print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | test_recon_mse~={test_mse:.6f} | {dt:.1f}s")

            if math.isnan(test_mse):
                test_mse = float("inf")

            if epoch == 1 or test_mse < best_test:
                best_test = test_mse
                torch.save(ckpt, ckpt_path)
                mlflow.log_artifact(os.path.join(out_dir, "conditional.pt"), artifact_path="checkpoints")
                mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")
                if os.path.exists(ckpt_path):
                    print(f"  ✓ NEW BEST: {os.path.abspath(ckpt_path)}")

    print("Done. Best approx test recon MSE:", best_test)
    last_ckpt_path = os.path.join(out_dir, "conditional.pt")
    torch.save(
        {
            "model": trainer.model.state_dict(),
            "cfg": cfg.__dict__,
            "train_mean": train_mean,
            "train_std": train_std,
        },
        last_ckpt_path,
    )
    return ckpt_path, (train_mean, train_std), cfg


@torch.no_grad()
def load_and_sample(
    ckpt_path: str,
    data,
    num_samples: int = 8,
    guidance_scale: float = 4.0,
):
    device = default_device()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = ckpt["cfg"]
    T = cfg_dict["T"]
    nt = cfg_dict.get("num_temporal", NUM_TEMPORAL)

    model = ConditionalDDPM3D(T=T, emb_dim=256, base_ch=64).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    cfg = DiffusionConfig(**cfg_dict)
    trainer = DDPMTrainer(model, cfg, device)

    if isinstance(data, np.ndarray):
        data_t = torch.from_numpy(data)
    else:
        data_t = data
    N = data_t.shape[0]
    n_train = int(0.8 * N)
    test_full = data_t[n_train:].float()
    mean = ckpt["train_mean"]
    std = ckpt["train_std"]

    idx = torch.randint(0, max(1, test_full.shape[0] - nt + 1), (num_samples,))
    coords = torch.arange(0, 64, 8)
    x0_list = []
    y_list = []
    for s in range(num_samples):
        i0 = idx[s].item()
        frames = []
        sparse = []
        for j in range(nt):
            fj = test_full[i0 + j]
            fn = (fj - mean) / (std + 1e-8)
            sparse.append(fn[coords][:, coords])
            frames.append(fj)
        x0_list.append(torch.stack(frames, dim=0))
        y_list.append(torch.stack(sparse, dim=0))
    x0 = torch.stack(x0_list, dim=0)
    yn = torch.stack(y_list, dim=0).unsqueeze(1).to(device)

    xhat = trainer.sample_cfg(
        y=yn,
        guidance_scale=guidance_scale,
        shape=(num_samples, 1, nt, 64, 64),
    )
    xhat = xhat.squeeze(1) * (std + 1e-8) + mean
    return x0.cpu(), xhat.cpu(), yn.squeeze(1).cpu()
