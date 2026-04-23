import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from interpolation import load_data, interpolate_sparse


class SparseEvalDataset(Dataset):
    def __init__(self, full_fields: torch.Tensor, mean: float, std: float, sensor_stride: int = 8):
        assert full_fields.ndim == 3 and full_fields.shape[1:] == (64, 64), (
            f"Expected (N,64,64), got {tuple(full_fields.shape)}"
        )
        self.x = full_fields.float()
        self.mean = float(mean)
        self.std = float(std)
        self.coords = torch.arange(0, 64, sensor_stride, dtype=torch.long)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x0 = self.x[idx]  # (64,64), original scale
        x0_norm = (x0 - self.mean) / (self.std + 1e-8)
        c = self.coords
        y_norm = x0_norm[c][:, c]  # (8,8)
        return x0_norm.unsqueeze(0), y_norm.unsqueeze(0)


@torch.no_grad()
def evaluate_split(
    loader: DataLoader,
    method: str,
    mean: float,
    std: float,
    device: torch.device,
) -> Dict[str, float]:
    total_pixels = 0
    mse_sum = 0.0
    mae_sum = 0.0
    total_samples = 0

    for x0_norm, y_norm in loader:
        x0_norm = x0_norm.to(device)
        y_norm = y_norm.to(device)

        xhat_norm = interpolate_sparse(y_norm, method=method)

        x0 = x0_norm * (std + 1e-8) + mean
        xhat = xhat_norm * (std + 1e-8) + mean

        diff = xhat - x0
        mse_sum += float((diff * diff).sum().item())
        mae_sum += float(diff.abs().sum().item())
        total_pixels += x0.numel()
        total_samples += x0.shape[0]

    return {
        "mse": mse_sum / max(total_pixels, 1),
        "mae": mae_sum / max(total_pixels, 1),
        "total_samples": total_samples,
    }


def sample_subset(x: torch.Tensor, n_samples: int, seed: int) -> torch.Tensor:
    n = x.shape[0]
    if n <= n_samples:
        return x
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=n_samples, replace=False)
    idx_t = torch.from_numpy(idx).long()
    return x[idx_t]


def run_eval(
    data_path: str,
    out_dir: str,
    method: str = "bilinear",
    n_samples_per_split: int = 100,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_full, test_full, train_mean, train_std = load_data(data_path)

    train_sampled = sample_subset(train_full, n_samples_per_split, seed=seed)
    test_sampled = sample_subset(test_full, n_samples_per_split, seed=seed + 1)

    train_ds = SparseEvalDataset(train_sampled, mean=train_mean, std=train_std, sensor_stride=8)
    test_ds = SparseEvalDataset(test_sampled, mean=train_mean, std=train_std, sensor_stride=8)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    print("=" * 60)
    print("Interpolation Evaluation (Random 100 Train + 100 Test)")
    print("=" * 60)
    print(f"Data path: {data_path}")
    print(f"Method: {method}")
    print(f"Device: {device}")
    print(f"Train sampled: {len(train_ds)} | Test sampled: {len(test_ds)}")
    print(f"Train mean={train_mean:.6f}, std={train_std:.6f}")
    print("=" * 60)

    train_metrics = evaluate_split(train_loader, method=method, mean=train_mean, std=train_std, device=device)
    test_metrics = evaluate_split(test_loader, method=method, mean=train_mean, std=train_std, device=device)

    total_samples = int(train_metrics["total_samples"]) + int(test_metrics["total_samples"])
    combined_mse = (
        float(train_metrics["mse"]) * int(train_metrics["total_samples"])
        + float(test_metrics["mse"]) * int(test_metrics["total_samples"])
    ) / max(total_samples, 1)
    combined_mae = (
        float(train_metrics["mae"]) * int(train_metrics["total_samples"])
        + float(test_metrics["mae"]) * int(test_metrics["total_samples"])
    ) / max(total_samples, 1)

    summary = {
        "data_path": data_path,
        "method": method,
        "n_samples_per_split": n_samples_per_split,
        "seed": seed,
        "train": train_metrics,
        "test": test_metrics,
        "combined": {
            "mse": combined_mse,
            "mae": combined_mae,
            "total_samples": total_samples,
        },
    }

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "interpolation_eval_100x2.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nFinal Results")
    print("-" * 60)
    print(f" train: MSE={train_metrics['mse']:.6e} | MAE={train_metrics['mae']:.6e}")
    print(f"  test: MSE={test_metrics['mse']:.6e} | MAE={test_metrics['mae']:.6e}")
    print(f"combined: MSE={combined_mse:.6e} | MAE={combined_mae:.6e}")
    print(f"\nSaved: {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate interpolation on random train/test subsets.")
    p.add_argument(
        "--data_path",
        type=str,
        default="/Users/eugenekim/2dNS_Conditional_Diffusion/NSE_Data(Noisy).npy",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="/Users/eugenekim/2dNS_Conditional_Diffusion/Interpolation/results",
    )
    p.add_argument(
        "--method",
        type=str,
        default="bilinear",
        choices=["nearest", "bilinear", "bicubic", "linear", "cubic"],
    )
    p.add_argument("--n_samples_per_split", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    run_eval(
        data_path="/content/drive/MyDrive/Lab/CondDiff/NSE_Data(Noisy).npy",
        out_dir="/content/drive/MyDrive/Lab/CondDiff/Interpolation/results",
        method="bilinear",
        n_samples_per_split=100,
        batch_size=32,
        num_workers=0,
        seed=42,
    )
