"""
PIDM evaluation: same procedure as ddpm_sparse_cfg/evaluate.py (CFG sampling,
train/test subsplits, full-field and sensor MSE/MAE), but uses PIDM training
code in cfgConditional.py — 8×8 sparse conditioning and checkpoints that may
include extra cfg keys (e.g. lambda_phys). Physics loss is training-only.
"""
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cfgConditional import (
    ConditionalDDPM,
    DDPMTrainer,
    DiffusionConfig,
    NavierStokesSparseDataset,
    default_device,
)


def _diffusion_config_from_ckpt(cfg_dict: dict) -> DiffusionConfig:
    fields = set(DiffusionConfig.__dataclass_fields__.keys())
    return DiffusionConfig(**{k: v for k, v in cfg_dict.items() if k in fields})


@torch.no_grad()
def evaluate_on_test(
    trainer,
    test_loader,
    mean,
    std,
    num_batches=10,
    guidance_scale=4.0,
    sensor_stride: int = 8,
):
    """
    Runs CFG sampling and reports:
      - full-field MSE/MAE in original scale
      - sensor MSE in original scale (8×8 grid, stride sensor_stride)
    """
    device = trainer.device
    trainer.model.eval()

    coords = torch.arange(0, 64, sensor_stride, device=device)

    def H(x64):  # x64: (B,64,64)
        return x64[:, coords][:, :, coords]  # (B, S, S), S = 64 // sensor_stride

    mse_list, mae_list, sensor_mse_list = [], [], []

    print(f"\n{'='*60}")
    print(f"Starting Evaluation")
    print(f"{'='*60}")
    if num_batches is not None:
        print(f"Evaluating on {num_batches} batches")
    else:
        print(f"Evaluating on all batches")
    print(f"Guidance scale: {guidance_scale}")
    print(f"Sensor stride: {sensor_stride} (grid {len(coords)}×{len(coords)})")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    total_samples = 0
    start_time = time.time()

    for b, (x0_norm, y_norm) in enumerate(test_loader):
        if num_batches is not None and b >= num_batches:
            break

        batch_start = time.time()
        x0_norm = x0_norm.to(device)  # (B,1,64,64), normalized
        y_norm = y_norm.to(device)  # (B,1,8,8), normalized — matches PIDM CondEncoder8x8

        B = x0_norm.size(0)
        total_samples += B

        if num_batches is not None:
            print(f"  Batch {b+1}/{num_batches} | Batch size: {B} | Sampling...", end=" ", flush=True)
        else:
            print(f"  Batch {b+1}/{len(test_loader)} | Batch size: {B} | Sampling...", end=" ", flush=True)

        sample_start = time.time()
        xhat_norm = trainer.sample_cfg(
            y=y_norm,
            guidance_scale=guidance_scale,
            shape=(B, 1, 64, 64),
        )
        sample_time = time.time() - sample_start

        x0 = x0_norm.squeeze(1) * (std + 1e-8) + mean
        xhat = xhat_norm.squeeze(1) * (std + 1e-8) + mean

        mse = F.mse_loss(xhat, x0).item()
        mae = F.l1_loss(xhat, x0).item()

        y_true = H(x0)
        y_pred = H(xhat)
        sensor_mse = F.mse_loss(y_pred, y_true).item()

        mse_list.append(mse)
        mae_list.append(mae)
        sensor_mse_list.append(sensor_mse)

        _ = time.time() - batch_start
        print(f"✓ | MSE: {mse:.6f} | MAE: {mae:.6f} | Sensor MSE: {sensor_mse:.6f} | Time: {sample_time:.1f}s")

    total_time = time.time() - start_time

    avg_mse = float(np.mean(mse_list)) if mse_list else float("nan")
    avg_mae = float(np.mean(mae_list)) if mae_list else float("nan")
    avg_sensor_mse = float(np.mean(sensor_mse_list)) if sensor_mse_list else float("nan")

    print(f"\n{'='*60}")
    print(f"Evaluation Complete")
    print(f"{'='*60}")
    print(f"Total batches evaluated: {len(mse_list)}")
    print(f"Total samples evaluated: {total_samples}")
    print(f"Total time: {total_time:.1f}s ({total_time/total_samples:.2f}s per sample)")
    print(f"\nAverage Metrics:")
    print(f"  Full-field MSE:  {avg_mse:.6f}")
    print(f"  Full-field MAE:  {avg_mae:.6f}")
    print(f"  Sensor MSE:      {avg_sensor_mse:.6f}")
    print(f"{'='*60}\n")

    return {
        "mse": avg_mse,
        "mae": avg_mae,
        "sensor_mse": avg_sensor_mse,
        "num_batches": num_batches,
        "guidance_scale": guidance_scale,
        "total_samples": total_samples,
        "total_time": total_time,
    }


def run_eval(
    ckpt_path: str,
    data_path: str,
    batch_size: int = 32,
    num_batches: int = 10,
    guidance_scale: float = None,
    sensor_stride: int = 8,
):
    device = default_device()
    print("Device:", device)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint file not found: {ckpt_path}\n"
            f"Train PIDM (cfgConditional.run_training / resume) to produce a .pt checkpoint."
        )

    if not ckpt_path.endswith(".pt"):
        raise ValueError(
            f"Checkpoint file should be a .pt file (PyTorch checkpoint), got: {ckpt_path}\n"
        )

    print(f"\n{'='*60}")
    print(f"Loading Data")
    print(f"{'='*60}")
    print(f"Data path: {data_path}")
    data = np.load(data_path).astype(np.float32)
    data_t = torch.from_numpy(data)
    N = data_t.shape[0]
    n_train = int(0.8 * N)
    n_test = N - n_train

    train_full = data_t[:n_train]
    test_full = data_t[n_train:]

    print(f"Total samples: {N}")
    print(f"Train samples: {n_train} (80%)")
    print(f"Test samples: {n_test} (20%)")
    print(f"Data shape: {data.shape}")

    n_samples_per_split = 100
    np.random.seed(42)

    if n_train < n_samples_per_split:
        print(f"Warning: Only {n_train} train samples available, using all of them")
        train_indices = np.arange(n_train)
    else:
        train_indices = np.random.choice(n_train, n_samples_per_split, replace=False)

    if n_test < n_samples_per_split:
        print(f"Warning: Only {n_test} test samples available, using all of them")
        test_indices = np.arange(n_test)
    else:
        test_indices = np.random.choice(n_test, n_samples_per_split, replace=False)

    train_sampled = train_full[train_indices]
    test_sampled = test_full[test_indices]
    eval_data = torch.cat([train_sampled, test_sampled], dim=0)

    print(f"\nSampling for evaluation:")
    print(f"  Train samples selected: {len(train_indices)} (from {n_train} available)")
    print(f"  Test samples selected: {len(test_indices)} (from {n_test} available)")
    print(f"  Total evaluation samples: {len(eval_data)}")

    print(f"\n{'='*60}")
    print(f"Loading Checkpoint (PIDM)")
    print(f"{'='*60}")
    print(f"Checkpoint path: {ckpt_path}")
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        print("✓ Checkpoint loaded successfully")
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
        print("✓ Checkpoint loaded successfully")
    except Exception as e:
        raise RuntimeError(
            f"Failed to load checkpoint: {e}. Make sure the file is a valid PyTorch checkpoint (.pt file)."
        )

    mean = float(ckpt["train_mean"])
    std = float(ckpt["train_std"])
    cfg_dict = ckpt["cfg"]
    checkpoint_guidance_scale = cfg_dict.get("guidance_scale", 4.0)

    print(f"Train mean: {mean:.6f}")
    print(f"Train std: {std:.6f}")
    print(f"Model T (timesteps): {cfg_dict.get('T', 'N/A')}")
    print(f"Checkpoint guidance_scale: {checkpoint_guidance_scale}")

    if guidance_scale is None:
        guidance_scale = checkpoint_guidance_scale
        print(f"Using checkpoint guidance_scale: {guidance_scale}")
    else:
        print(
            f"Using explicit guidance_scale: {guidance_scale} (checkpoint default was {checkpoint_guidance_scale})"
        )

    print(f"\n{'='*60}")
    print(f"Building Model")
    print(f"{'='*60}")
    model = ConditionalDDPM(T=cfg_dict["T"], emb_dim=256, base_ch=64).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print("✓ Model loaded and set to eval mode")

    cfg = _diffusion_config_from_ckpt(cfg_dict)
    trainer = DDPMTrainer(model, cfg, device)
    print("✓ Trainer initialized")

    print(f"\n{'='*60}")
    print(f"Preparing Evaluation DataLoaders (sensor_stride={sensor_stride})")
    print(f"{'='*60}")

    train_ds = NavierStokesSparseDataset(train_sampled, mean=mean, std=std, sensor_stride=sensor_stride)
    test_ds = NavierStokesSparseDataset(test_sampled, mean=mean, std=std, sensor_stride=sensor_stride)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    print(f"Train dataset size: {len(train_ds)}")
    print(f"Test dataset size: {len(test_ds)}")
    print(f"Batch size: {batch_size}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    print(f"\n{'='*60}")
    print(f"EVALUATING TRAIN SAMPLES")
    print(f"{'='*60}")
    train_metrics = evaluate_on_test(
        trainer,
        train_loader,
        mean=mean,
        std=std,
        num_batches=None,
        guidance_scale=guidance_scale,
        sensor_stride=sensor_stride,
    )

    print(f"\n{'='*60}")
    print(f"EVALUATING TEST SAMPLES")
    print(f"{'='*60}")
    test_metrics = evaluate_on_test(
        trainer,
        test_loader,
        mean=mean,
        std=std,
        num_batches=None,
        guidance_scale=guidance_scale,
        sensor_stride=sensor_stride,
    )

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS SUMMARY (PIDM)")
    print(f"{'='*60}")
    print(f"\nTRAIN SET METRICS ({train_metrics['total_samples']} samples):")
    print(f"  Full-field MSE:  {train_metrics['mse']:.6f}")
    print(f"  Full-field MAE:  {train_metrics['mae']:.6f}")
    print(f"  Sensor MSE:      {train_metrics['sensor_mse']:.6f}")
    print(f"  Total time:      {train_metrics['total_time']:.1f}s")

    print(f"\nTEST SET METRICS ({test_metrics['total_samples']} samples):")
    print(f"  Full-field MSE:  {test_metrics['mse']:.6f}")
    print(f"  Full-field MAE:  {test_metrics['mae']:.6f}")
    print(f"  Sensor MSE:      {test_metrics['sensor_mse']:.6f}")
    print(f"  Total time:      {test_metrics['total_time']:.1f}s")

    total_samples = train_metrics["total_samples"] + test_metrics["total_samples"]
    combined_mse = (
        train_metrics["mse"] * train_metrics["total_samples"]
        + test_metrics["mse"] * test_metrics["total_samples"]
    ) / total_samples
    combined_mae = (
        train_metrics["mae"] * train_metrics["total_samples"]
        + test_metrics["mae"] * test_metrics["total_samples"]
    ) / total_samples
    combined_sensor_mse = (
        train_metrics["sensor_mse"] * train_metrics["total_samples"]
        + test_metrics["sensor_mse"] * test_metrics["total_samples"]
    ) / total_samples

    print(f"\nCOMBINED METRICS ({total_samples} samples):")
    print(f"  Full-field MSE:  {combined_mse:.6f}")
    print(f"  Full-field MAE:  {combined_mae:.6f}")
    print(f"  Sensor MSE:      {combined_sensor_mse:.6f}")
    print(f"  Total time:      {train_metrics['total_time'] + test_metrics['total_time']:.1f}s")
    print(f"  Guidance scale:  {guidance_scale}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    _SRC = Path(__file__).resolve().parent
    _PIDM_ROOT = _SRC.parent
    _PROJ = _PIDM_ROOT.parent

    default_ckpt = str(_PROJ / "checkpoint" / "best.pt")
    if not os.path.isfile(default_ckpt):
        _alt = _PIDM_ROOT / "trained_models" / "run_1" / "model" / "checkpoint_0.pt"
        if _alt.is_file():
            default_ckpt = str(_alt)

    data_path = str(_PROJ / "NSE_Data(Noisy).npy")

    if not os.path.exists(default_ckpt):
        print("=" * 60)
        print("ERROR: No PIDM checkpoint found!")
        print("=" * 60)
        print(f"Tried: {os.path.abspath(default_ckpt)}")
        print("\nTrain PIDM or set a path to your .pt file, then run:")
        print(f"  python -m evaluate  # from {_SRC}")
        print("or:")
        print("  python evaluate.py   # with cwd = PIDM/src")
        print("=" * 60)
        raise SystemExit(1)

    run_eval(
        ckpt_path=default_ckpt,
        data_path=data_path,
        batch_size=16,
        num_batches=5,
        guidance_scale=None,
        sensor_stride=8,
    )
