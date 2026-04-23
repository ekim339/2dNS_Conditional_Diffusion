import os
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

_PIDM_ROOT = Path(__file__).resolve().parent
_SRC = _PIDM_ROOT / "src"
sys.path.insert(0, str(_SRC))

from cfgConditional import (
    ConditionalDDPM, DDPMTrainer, DiffusionConfig,
    default_device
)

def _diffusion_config_from_ckpt(cfg_dict: dict) -> DiffusionConfig:
    fields = set(DiffusionConfig.__dataclass_fields__.keys())
    return DiffusionConfig(**{k: v for k, v in cfg_dict.items() if k in fields})

def generate_and_plot_sample(
    trainer,
    x0_true: torch.Tensor,
    mean: float,
    std: float,
    device: torch.device,
    guidance_scale: float = 4.0,
    title_prefix: str = "",
):
    """
    One CFG/DDPM sample from sparse observations vs ground truth.

    Returns:
        x_true, x_pred, diff (numpy), mse, mae
    """
    # Ensure x0_true is 2D (64, 64)
    if x0_true.dim() > 2:
        x0_true = x0_true.squeeze()
    assert x0_true.shape == (64, 64), f"Expected (64, 64), got {x0_true.shape}"
    
    # Normalize
    x0_true_norm = (x0_true - mean) / (std + 1e-8)
    
    # Build sparse observation y (8x8) for PIDM.
    coords = torch.arange(0, 64, 8, dtype=torch.long)
    c = coords
    y_sparse = x0_true_norm[c][:, c]  # (8, 8)
    
    # Verify sparse observation shape and values
    assert y_sparse.shape == (8, 8), f"Expected y_sparse shape (8, 8), got {y_sparse.shape}"
    
    # Prepare for model input
    y_input = y_sparse.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 8, 8)
    assert y_input.shape == (1, 1, 8, 8), f"Expected y_input shape (1, 1, 8, 8), got {y_input.shape}"
    
    print(f"Generating single sample ({title_prefix})...", end=" ", flush=True)
    with torch.no_grad():
        x_pred_norm = trainer.sample_cfg(
            y=y_input,
            guidance_scale=guidance_scale,
            shape=(1, 1, 64, 64),
        )
        if x_pred_norm.dim() == 4:
            x_pred_norm = x_pred_norm.squeeze()
        x_pred_t = x_pred_norm.cpu() * (std + 1e-8) + mean # (64, 64)
    print("Done.")

    x_true = x0_true.cpu().numpy()
    x_pred = x_pred_t.numpy()
    diff = x_true - x_pred
    
    # Compute metrics
    mse = np.mean((x_true - x_pred) ** 2)
    mae = np.mean(np.abs(x_true - x_pred))
    
    return x_true, x_pred, diff, mse, mae


def _add_sparse_grid_lines(ax, stride: int = 8, n: int = 64):
    """Dashed lines for coarse 8x8 sensor grid on 64x64 fields."""
    for i in range(1, n // stride):
        ax.axvline(x=i * stride - 0.5, color="white", linewidth=0.8, alpha=0.55, linestyle="--")
        ax.axhline(y=i * stride - 0.5, color="white", linewidth=0.8, alpha=0.55, linestyle="--")


def _plot_split_grid(rows, split_name: str, save_path: str):
    """
    Save a 3x3 figure:
      rows = samples (3 indices), cols = [original, generated, difference]
    """
    nrows = len(rows)
    fig, axes = plt.subplots(nrows, 3, figsize=(14, 4.0 * nrows))
    axes = np.atleast_2d(axes)

    for r, row in enumerate(rows):
        xt = row["x_true"]
        xp = row["x_pred"]
        xd = row["diff"]
        idx = row["index"]

        vmin = float(min(xt.min(), xp.min()))
        vmax = float(max(xt.max(), xp.max()))
        dmax = float(np.abs(xd).max())

        # Original
        ax0 = axes[r, 0]
        im0 = ax0.imshow(xt, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
        ax0.set_title(f"{split_name} idx {idx}\nOriginal", fontsize=10, fontweight="bold")
        ax0.axis("off")
        _add_sparse_grid_lines(ax0)
        plt.colorbar(im0, ax=ax0, fraction=0.046)

        # Generated
        ax1 = axes[r, 1]
        im1 = ax1.imshow(xp, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
        ax1.set_title(
            f"Generated\nMSE={row['mse']:.4e}  MAE={row['mae']:.4e}",
            fontsize=10,
            fontweight="bold",
        )
        ax1.axis("off")
        _add_sparse_grid_lines(ax1)
        plt.colorbar(im1, ax=ax1, fraction=0.046)

        # Difference
        ax2 = axes[r, 2]
        im2 = ax2.imshow(xd, cmap="RdBu_r", aspect="auto", vmin=-dmax, vmax=dmax)
        ax2.set_title("Difference (true - gen)", fontsize=10, fontweight="bold")
        ax2.axis("off")
        _add_sparse_grid_lines(ax2)
        plt.colorbar(im2, ax=ax2, fraction=0.046)

    fig.suptitle(f"{split_name.capitalize()} samples: Original vs Generated vs Difference", fontsize=13, y=1.002)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {split_name} figure to: {save_path}")


def compare_train_and_test_samples(
    ckpt_path: str,
    data_path: str,
    guidance_scale: float = None,  # None = use checkpoint default
    save_path: str = None,
    train_indices=None,
    test_indices=None,
):
    """
    Generate and visualize multiple train and test fields.

    Default: three consecutive train indices [49999, 50000, 50001] and
    three consecutive test indices [0, 1, 2]. Each row shows Original | Generated side by side.
    """
    if train_indices is None:
        train_indices = [49999, 50000, 50001]
    if test_indices is None:
        test_indices = [0, 1, 2]

    device = default_device()
    
    # Load data
    print(f"Loading data from: {data_path}")
    data = np.load(data_path).astype(np.float32)
    print(f"Data shape: {data.shape}")
    
    # Split into train/test
    data_t = torch.from_numpy(data)
    N = data_t.shape[0]
    n_train = int(0.8 * N)
    
    train_full = data_t[:n_train].float()  # first 80% is train
    test_full = data_t[n_train:].float()  # last 20% is test
    
    print(f"Total samples: {N}")
    print(f"Train samples: {n_train} (80%)")
    print(f"Test samples: {len(test_full)} (20%)")
    
    # Load checkpoint
    print(f"\nLoading checkpoint from: {ckpt_path}")
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    mean = float(ckpt["train_mean"])
    std = float(ckpt["train_std"])
    cfg_dict = ckpt["cfg"]
    checkpoint_guidance_scale = cfg_dict.get('guidance_scale', 4.0)
    
    print(f"Train mean: {mean:.6f}")
    print(f"Train std: {std:.6f}")
    print(f"Checkpoint guidance_scale: {checkpoint_guidance_scale}")
    
    # Use checkpoint guidance_scale if not explicitly provided
    if guidance_scale is None:
        guidance_scale = checkpoint_guidance_scale
        print(f"Using checkpoint guidance_scale: {guidance_scale}")
    else:
        print(f"Using explicit guidance_scale: {guidance_scale} (checkpoint default was {checkpoint_guidance_scale})")
    
    # Rebuild model
    model = ConditionalDDPM(T=cfg_dict["T"], emb_dim=256, base_ch=64).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    
    cfg = _diffusion_config_from_ckpt(cfg_dict)
    trainer = DDPMTrainer(model, cfg, device)

    train_rows = []
    for idx in train_indices:
        if idx < 0 or idx >= len(train_full):
            raise IndexError(f"Train index {idx} out of range [0, {len(train_full) - 1}]")
        print(f"\n{'='*60}\nTrain index {idx}\n{'='*60}")
        x_true, x_pred, diff, mse, mae = generate_and_plot_sample(
            trainer,
            train_full[idx],
            mean,
            std,
            device,
            guidance_scale,
            f"Train[{idx}]",
        )
        train_rows.append(
            {
                "split": "train",
                "index": idx,
                "x_true": x_true,
                "x_pred": x_pred,
                "diff": diff,
                "mse": mse,
                "mae": mae,
            }
        )
        print(f"  MSE: {mse:.6f}, MAE: {mae:.6f}")

    test_rows = []
    for idx in test_indices:
        if idx < 0 or idx >= len(test_full):
            raise IndexError(f"Test index {idx} out of range [0, {len(test_full) - 1}]")
        print(f"\n{'='*60}\nTest index {idx} (within test split)\n{'='*60}")
        x_true, x_pred, diff, mse, mae = generate_and_plot_sample(
            trainer,
            test_full[idx],
            mean,
            std,
            device,
            guidance_scale,
            f"Test[{idx}]",
        )
        test_rows.append(
            {
                "split": "test",
                "index": idx,
                "x_true": x_true,
                "x_pred": x_pred,
                "diff": diff,
                "mse": mse,
                "mae": mae,
            }
        )
        print(f"  MSE: {mse:.6f}, MAE: {mae:.6f}")

    samples_dir = str("/Users/eugenekim/2dNS_Conditional_Diffusion/PIDM/samples")
    os.makedirs(samples_dir, exist_ok=True)
    train_save = os.path.join(samples_dir, "train_samples_3x3_pidm.png")
    test_save = os.path.join(samples_dir, "test_samples_3x3_pidm.png")

    _plot_split_grid(train_rows, "train", train_save)
    _plot_split_grid(test_rows, "test", test_save)
    
    all_rows = train_rows + test_rows
    print(f"\n{'='*60}\nSummary ({len(all_rows)} samples)\n{'='*60}")
    for row in all_rows:
        print(
            f"  {row['split']} idx {row['index']:5d} | "
            f"MSE: {row['mse']:.6f} | MAE: {row['mae']:.6f}"
        )
    print(f"{'='*60}\n")

if __name__ == "__main__":
    #Configuration
    ckpt_path = "/Users/eugenekim/2dNS_Conditional_Diffusion/checkpoint/PIDM_low_freq.pt"
    data_path = "/Users/eugenekim/2dNS_Conditional_Diffusion/NSE_Data(Noisy).npy"
    guidance_scale = None  # None = use checkpoint's guidance_scale, or set explicitly (e.g., 4.0)
    
    compare_train_and_test_samples(
        ckpt_path=ckpt_path,
        data_path=data_path,
        guidance_scale=guidance_scale,
        save_path=None,  # e.g. "comparison.png"
        train_indices=[49999, 50000, 50001],
        test_indices=[0, 1, 2],
    )
