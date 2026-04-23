"""
Direct spatial interpolation baseline for sparse 8x8 -> full 64x64 2D Navier-Stokes fields.

This script mirrors the data convention in cfgConditional.py:
  - data is a NumPy array with shape (N, 64, 64)
  - first 80% is train, last 20% is test
  - normalization statistics are computed from train only
  - sparse observations are sampled at pixels [0, 8, ..., 56] on both axes

The interpolation itself is deterministic and has no learned parameters.
"""

import argparse
import json
import os
import time
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


SENSOR_STRIDE = 8
FIELD_SIZE = 64
SENSOR_SIZE = 8


class NavierStokesSparseDataset(Dataset):
    def __init__(self, full_fields: torch.Tensor, mean: float, std: float, sensor_stride: int = SENSOR_STRIDE):
        assert full_fields.ndim == 3 and full_fields.shape[1:] == (FIELD_SIZE, FIELD_SIZE)
        self.x = full_fields.float()
        self.mean = float(mean)
        self.std = float(std)
        self.sensor_stride = int(sensor_stride)

        coords = torch.arange(0, FIELD_SIZE, self.sensor_stride)
        assert len(coords) == SENSOR_SIZE, "Expected exactly 8 points per axis for 8x8 sensors."
        self.registered_coords = coords

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        x0 = self.x[idx]
        x0 = (x0 - self.mean) / (self.std + 1e-8)

        c = self.registered_coords
        y = x0[c][:, c]

        return x0.unsqueeze(0), y.unsqueeze(0)


def load_data(data_path: str) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    data = np.load(data_path).astype(np.float32)
    data_t = torch.from_numpy(data)
    assert data_t.ndim == 3 and data_t.shape[1:] == (FIELD_SIZE, FIELD_SIZE), (
        f"Expected data shape (N,64,64), got {tuple(data_t.shape)}"
    )

    n_train = int(0.8 * data_t.shape[0])
    train_full = data_t[:n_train]
    test_full = data_t[n_train:]

    train_mean = train_full.float().mean().item()
    train_std = train_full.float().std().item()
    return train_full, test_full, train_mean, train_std


def make_loaders(
    data_path: str,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, Dict[str, float]]:
    train_full, test_full, train_mean, train_std = load_data(data_path)

    train_ds = NavierStokesSparseDataset(train_full, mean=train_mean, std=train_std)
    test_ds = NavierStokesSparseDataset(test_full, mean=train_mean, std=train_std)

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

    info = {
        "num_total": len(train_ds) + len(test_ds),
        "num_train": len(train_ds),
        "num_test": len(test_ds),
        "train_mean": train_mean,
        "train_std": train_std,
    }
    return train_loader, test_loader, info


def _linear_extrapolate_9x9(y8: torch.Tensor) -> torch.Tensor:
    """
    Extend 8x8 sensor values at coordinates 0,8,...,56 to a 9x9 grid at 0,8,...,64.

    This gives bilinear interpolation a proper right/bottom boundary for reconstructing
    pixels 57..63, instead of clamping or treating the last observed sensor as the image edge.
    """
    y9 = F.pad(y8, (0, 1, 0, 1), mode="replicate")
    y9[..., :8, 8] = 2.0 * y8[..., :8, 7] - y8[..., :8, 6]
    y9[..., 8, :8] = 2.0 * y8[..., 7, :8] - y8[..., 6, :8]
    y9[..., 8, 8] = 2.0 * y9[..., 8, 7] - y9[..., 8, 6]
    return y9


def interpolate_torch(y: torch.Tensor, method: str) -> torch.Tensor:
    if method == "nearest":
        return F.interpolate(y, size=(FIELD_SIZE, FIELD_SIZE), mode="nearest-exact")

    if method == "bilinear":
        y9 = _linear_extrapolate_9x9(y)
        x65 = F.interpolate(y9, size=(FIELD_SIZE + 1, FIELD_SIZE + 1), mode="bilinear", align_corners=True)
        return x65[..., :FIELD_SIZE, :FIELD_SIZE]

    if method == "bicubic":
        y9 = _linear_extrapolate_9x9(y)
        x65 = F.interpolate(y9, size=(FIELD_SIZE + 1, FIELD_SIZE + 1), mode="bicubic", align_corners=True)
        return x65[..., :FIELD_SIZE, :FIELD_SIZE]

    raise ValueError(f"Unknown interpolation method: {method}")


def _try_scipy_griddata(
    y: torch.Tensor,
    method: str,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if method not in {"linear", "cubic"}:
        return None

    try:
        from scipy.interpolate import griddata
    except ImportError:
        return None

    y_np = y.detach().cpu().numpy()
    sensor_coords = np.arange(0, FIELD_SIZE, SENSOR_STRIDE, dtype=np.float32)
    yy, xx = np.meshgrid(sensor_coords, sensor_coords, indexing="ij")
    points = np.stack([yy.ravel(), xx.ravel()], axis=1)

    full_coords = np.arange(FIELD_SIZE, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(full_coords, full_coords, indexing="ij")

    out = np.empty((y_np.shape[0], 1, FIELD_SIZE, FIELD_SIZE), dtype=np.float32)
    for i in range(y_np.shape[0]):
        values = y_np[i, 0].reshape(-1)
        interp = griddata(points, values, (grid_y, grid_x), method=method)

        # Cubic/linear griddata returns NaN outside the convex hull, i.e. the 57..63 boundary.
        # Fill those pixels with nearest-neighbor extrapolation so metrics remain finite.
        if np.isnan(interp).any():
            nearest = griddata(points, values, (grid_y, grid_x), method="nearest")
            interp = np.where(np.isnan(interp), nearest, interp)
        out[i, 0] = interp.astype(np.float32)

    return torch.from_numpy(out).to(device)


def interpolate_sparse(y: torch.Tensor, method: str) -> torch.Tensor:
    """
    Reconstruct normalized full fields from normalized sparse observations.

    Supported methods:
      nearest: nearest upsampling
      bilinear: torch bilinear interpolation with linear right/bottom extrapolation
      bicubic: torch bicubic interpolation with linear right/bottom extrapolation
      linear: scipy griddata linear interpolation if scipy is installed, otherwise bilinear fallback
      cubic: scipy griddata cubic interpolation if scipy is installed, otherwise bicubic fallback
    """
    method = method.lower()
    scipy_out = _try_scipy_griddata(y, method, y.device)
    if scipy_out is not None:
        return scipy_out

    fallback = {"linear": "bilinear", "cubic": "bicubic"}.get(method, method)
    return interpolate_torch(y, fallback)


def mse_mae_sum(pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
    diff = pred - target
    return float((diff * diff).sum().item()), float(diff.abs().sum().item())


@torch.no_grad()
def evaluate_loader(
    loader: DataLoader,
    method: str,
    mean: float,
    std: float,
    device: torch.device,
    max_batches: Optional[int] = None,
    save_predictions: bool = False,
) -> Dict[str, object]:
    total_samples = 0
    total_pixels = 0
    mse_total = 0.0
    mae_total = 0.0
    sensor_mse_total = 0.0
    predictions = []
    targets = []

    coords = torch.arange(0, FIELD_SIZE, SENSOR_STRIDE, device=device)
    start_time = time.time()

    for batch_idx, (x0_norm, y_norm) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x0_norm = x0_norm.to(device)
        y_norm = y_norm.to(device)
        xhat_norm = interpolate_sparse(y_norm, method=method)

        x0 = x0_norm * (std + 1e-8) + mean
        xhat = xhat_norm * (std + 1e-8) + mean

        batch_samples = x0.shape[0]
        batch_pixels = x0.numel()
        total_samples += batch_samples
        total_pixels += batch_pixels

        batch_mse_sum, batch_mae_sum = mse_mae_sum(xhat, x0)
        mse_total += batch_mse_sum
        mae_total += batch_mae_sum

        sensor_true = x0[:, :, coords][:, :, :, coords]
        sensor_pred = xhat[:, :, coords][:, :, :, coords]
        sensor_mse_total += float(((sensor_pred - sensor_true) ** 2).sum().item())

        if save_predictions:
            predictions.append(xhat.cpu())
            targets.append(x0.cpu())

    elapsed = time.time() - start_time
    metrics: Dict[str, object] = {
        "method": method,
        "mse": mse_total / max(total_pixels, 1),
        "mae": mae_total / max(total_pixels, 1),
        "sensor_mse": sensor_mse_total / max(total_samples * SENSOR_SIZE * SENSOR_SIZE, 1),
        "total_samples": total_samples,
        "total_time": elapsed,
        "seconds_per_sample": elapsed / max(total_samples, 1),
    }

    if save_predictions:
        metrics["predictions"] = torch.cat(predictions, dim=0).numpy() if predictions else np.empty((0, 1, 64, 64))
        metrics["targets"] = torch.cat(targets, dim=0).numpy() if targets else np.empty((0, 1, 64, 64))

    return metrics


def strip_arrays(metrics: Dict[str, object]) -> Dict[str, object]:
    return {k: v for k, v in metrics.items() if k not in {"predictions", "targets"}}


def write_metrics(out_dir: str, summary: Dict[str, object]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, "direct_spatial_interpolation_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return metrics_path


def save_npz(out_dir: str, split: str, metrics: Dict[str, object]) -> Optional[str]:
    if "predictions" not in metrics:
        return None
    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, f"direct_spatial_interpolation_{split}.npz")
    np.savez_compressed(
        npz_path,
        predictions=metrics["predictions"],
        targets=metrics["targets"],
    )
    return npz_path


def _default_resume_state_path(out_dir: str) -> str:
    return os.path.join(out_dir, "interpolation_resume_state.json")


def _load_resume_state(path: str) -> Dict[str, object]:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_resume_state(path: str, state: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def run_interpolation(
    data_path: str,
    out_dir: str,
    method: str = "bilinear",
    batch_size: int = 256,
    num_workers: int = 0,
    max_batches: Optional[int] = None,
    save_predictions: bool = False,
    epochs: int = 1,
    resume: bool = False,
    resume_state_path: Optional[str] = None,
) -> Dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, info = make_loaders(data_path, batch_size=batch_size, num_workers=num_workers)
    os.makedirs(out_dir, exist_ok=True)
    resume_state_path = resume_state_path or _default_resume_state_path(out_dir)

    start_epoch = 0
    if resume:
        state = _load_resume_state(resume_state_path)
        start_epoch = int(state.get("last_epoch", 0))
        print(f"Resume enabled. State file: {resume_state_path}")
        print(f"Last completed epoch from state: {start_epoch}")

    print("=" * 60)
    print("Direct Spatial Interpolation Baseline")
    print("=" * 60)
    print(f"Data path: {data_path}")
    print(f"Method: {method}")
    print(f"Device: {device}")
    print(f"Total samples: {int(info['num_total'])}")
    print(f"Train samples: {int(info['num_train'])}")
    print(f"Test samples: {int(info['num_test'])}")
    print(f"Train mean={info['train_mean']:.6f}, std={info['train_std']:.6f}")
    print(f"Sensors: 8x8 at indices {list(range(0, FIELD_SIZE, SENSOR_STRIDE))}")
    print(f"Epochs: {epochs} (start from {start_epoch + 1})")
    print("=" * 60)
    summary: Dict[str, object] = {}
    if start_epoch >= epochs:
        print(f"All requested epochs already completed (last={start_epoch}, requested={epochs}).")
        latest_metrics = os.path.join(out_dir, "direct_spatial_interpolation_metrics.json")
        if os.path.isfile(latest_metrics):
            with open(latest_metrics, "r", encoding="utf-8") as f:
                return json.load(f)
        return summary

    for epoch in range(start_epoch + 1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        train_metrics = evaluate_loader(
            train_loader,
            method=method,
            mean=float(info["train_mean"]),
            std=float(info["train_std"]),
            device=device,
            max_batches=max_batches,
            save_predictions=save_predictions,
        )
        test_metrics = evaluate_loader(
            test_loader,
            method=method,
            mean=float(info["train_mean"]),
            std=float(info["train_std"]),
            device=device,
            max_batches=max_batches,
            save_predictions=save_predictions,
        )

        total_samples = int(train_metrics["total_samples"]) + int(test_metrics["total_samples"])
        combined = {
            "method": method,
            "mse": (
                float(train_metrics["mse"]) * int(train_metrics["total_samples"])
                + float(test_metrics["mse"]) * int(test_metrics["total_samples"])
            )
            / max(total_samples, 1),
            "mae": (
                float(train_metrics["mae"]) * int(train_metrics["total_samples"])
                + float(test_metrics["mae"]) * int(test_metrics["total_samples"])
            )
            / max(total_samples, 1),
            "sensor_mse": (
                float(train_metrics["sensor_mse"]) * int(train_metrics["total_samples"])
                + float(test_metrics["sensor_mse"]) * int(test_metrics["total_samples"])
            )
            / max(total_samples, 1),
            "total_samples": total_samples,
            "total_time": float(train_metrics["total_time"]) + float(test_metrics["total_time"]),
        }

        summary = {
            "data_path": data_path,
            "sensor_stride": SENSOR_STRIDE,
            "sensor_indices": list(range(0, FIELD_SIZE, SENSOR_STRIDE)),
            "normalization": {
                "train_mean": float(info["train_mean"]),
                "train_std": float(info["train_std"]),
            },
            "epoch": epoch,
            "epochs_requested": epochs,
            "splits": {
                "train": strip_arrays(train_metrics),
                "test": strip_arrays(test_metrics),
                "combined": combined,
            },
        }

        metrics_path = write_metrics(out_dir, summary)
        train_npz = save_npz(out_dir, "train", train_metrics)
        test_npz = save_npz(out_dir, "test", test_metrics)

        print("\nResults")
        print("-" * 60)
        for split_name, metrics in (("train", train_metrics), ("test", test_metrics), ("combined", combined)):
            print(
                f"{split_name:>8}: "
                f"MSE={float(metrics['mse']):.6e} | "
                f"MAE={float(metrics['mae']):.6e} | "
                f"Sensor MSE={float(metrics['sensor_mse']):.6e} | "
                f"Samples={int(metrics['total_samples'])}"
            )
        print(f"\nSaved metrics: {metrics_path}")
        if train_npz:
            print(f"Saved train predictions: {train_npz}")
        if test_npz:
            print(f"Saved test predictions: {test_npz}")

        _save_resume_state(
            resume_state_path,
            {
                "last_epoch": epoch,
                "epochs_requested": epochs,
                "data_path": data_path,
                "out_dir": out_dir,
                "method": method,
                "batch_size": batch_size,
                "num_workers": num_workers,
                "max_batches": max_batches,
                "save_predictions": save_predictions,
            },
        )
        print(f"Saved resume state: {resume_state_path}")

    return summary


def parse_args(argv: Optional[Iterable[str]] = None):
    parser = argparse.ArgumentParser(description="Direct spatial interpolation baseline for 8x8 sparse 2D NS observations.")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/Users/eugenekim/2dNS_Conditional_Diffusion/NSE_Data(Noisy).npy",
        help="Path to .npy data with shape (N,64,64).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/Users/eugenekim/2dNS_Conditional_Diffusion/ddpm_sparse_cfg/interpolation_results",
        help="Directory for metrics and optional prediction files.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="bilinear",
        choices=["nearest", "bilinear", "bicubic", "linear", "cubic"],
        help="Interpolation method. linear/cubic use scipy griddata when available, otherwise torch fallback.",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=None, help="Optional quick-test limit per split.")
    parser.add_argument("--save_predictions", action="store_true", help="Save reconstructed and target fields as compressed NPZ.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to run for this baseline loop.")
    parser.add_argument("--resume", action="store_true", help="Resume from saved interpolation state file.")
    parser.add_argument(
        "--resume_state_path",
        type=str,
        default=None,
        help="Path to resume state JSON. Default: <out_dir>/interpolation_resume_state.json",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run_interpolation(
        data_path=args.data_path,
        out_dir=args.out_dir,
        method=args.method,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_batches=args.max_batches,
        save_predictions=args.save_predictions,
        epochs=args.epochs,
        resume=args.resume,
        resume_state_path=args.resume_state_path,
    )
