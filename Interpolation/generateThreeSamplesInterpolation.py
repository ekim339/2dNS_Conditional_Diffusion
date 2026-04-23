import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from interpolation import interpolate_sparse


def generate_and_plot_sample(
    x0_true: torch.Tensor,
    mean: float,
    std: float,
    method: str = "bilinear",
    title_prefix: str = "",
):
    """
    One interpolation reconstruction from sparse observations vs ground truth.

    Returns:
        x_true, x_pred, diff (numpy), mse, mae
    """
    if x0_true.dim() > 2:
        x0_true = x0_true.squeeze()
    assert x0_true.shape == (64, 64), f"Expected (64, 64), got {x0_true.shape}"

    # Normalize, then build sparse 8x8 observation
    x0_true_norm = (x0_true - mean) / (std + 1e-8)
    coords = torch.arange(0, 64, 8, dtype=torch.long)
    c = coords
    y_sparse = x0_true_norm[c][:, c]  # (8, 8)
    assert y_sparse.shape == (8, 8), f"Expected y_sparse shape (8, 8), got {y_sparse.shape}"

    y_input = y_sparse.unsqueeze(0).unsqueeze(0)  # (1,1,8,8)
    print(f"Reconstructing single sample ({title_prefix}) with {method}...", end=" ", flush=True)
    with torch.no_grad():
        x_pred_norm = interpolate_sparse(y_input, method=method).squeeze(0).squeeze(0)
        x_pred_t = x_pred_norm.cpu() * (std + 1e-8) + mean
    print("Done.")

    x_true = x0_true.cpu().numpy()
    x_pred = x_pred_t.numpy()
    diff = x_true - x_pred

    mse = np.mean((x_true - x_pred) ** 2)
    mae = np.mean(np.abs(x_true - x_pred))
    return x_true, x_pred, diff, mse, mae


def _add_sparse_grid_lines(ax, stride: int = 8, n: int = 64):
    for i in range(1, n // stride):
        ax.axvline(x=i * stride - 0.5, color="white", linewidth=0.8, alpha=0.55, linestyle="--")
        ax.axhline(y=i * stride - 0.5, color="white", linewidth=0.8, alpha=0.55, linestyle="--")


def _plot_split_grid(rows, split_name: str, save_path: str, method: str):
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

        ax0 = axes[r, 0]
        im0 = ax0.imshow(xt, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
        ax0.set_title(f"{split_name} idx {idx}\nOriginal", fontsize=10, fontweight="bold")
        ax0.axis("off")
        _add_sparse_grid_lines(ax0)
        plt.colorbar(im0, ax=ax0, fraction=0.046)

        ax1 = axes[r, 1]
        im1 = ax1.imshow(xp, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
        ax1.set_title(
            f"Reconstructed ({method})\nMSE={row['mse']:.4e}  MAE={row['mae']:.4e}",
            fontsize=10,
            fontweight="bold",
        )
        ax1.axis("off")
        _add_sparse_grid_lines(ax1)
        plt.colorbar(im1, ax=ax1, fraction=0.046)

        ax2 = axes[r, 2]
        im2 = ax2.imshow(xd, cmap="RdBu_r", aspect="auto", vmin=-dmax, vmax=dmax)
        ax2.set_title("Difference (true - recon)", fontsize=10, fontweight="bold")
        ax2.axis("off")
        _add_sparse_grid_lines(ax2)
        plt.colorbar(im2, ax=ax2, fraction=0.046)

    fig.suptitle(
        f"{split_name.capitalize()} samples: Original vs Interpolation Reconstruction vs Difference",
        fontsize=13,
        y=1.002,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {split_name} figure to: {save_path}")


def compare_train_and_test_samples(
    data_path: str,
    method: str = "bilinear",
    save_path: str = None,
    train_indices=None,
    test_indices=None,
):
    """
    Generate and visualize multiple train and test fields with interpolation.

    Default: three consecutive train indices [49999, 50000, 50001] and
    three consecutive test indices [0, 1, 2] (within test split).
    """
    if train_indices is None:
        train_indices = [49999, 50000, 50001]
    if test_indices is None:
        test_indices = [0, 1, 2]

    print(f"Loading data from: {data_path}")
    data = np.load(data_path).astype(np.float32)
    print(f"Data shape: {data.shape}")

    data_t = torch.from_numpy(data)
    N = data_t.shape[0]
    n_train = int(0.8 * N)

    train_full = data_t[:n_train].float()
    test_full = data_t[n_train:].float()

    mean = float(train_full.mean().item())
    std = float(train_full.std().item())

    print(f"Total samples: {N}")
    print(f"Train samples: {n_train} (80%)")
    print(f"Test samples: {len(test_full)} (20%)")
    print(f"Train mean: {mean:.6f}")
    print(f"Train std: {std:.6f}")
    print(f"Interpolation method: {method}")

    train_rows = []
    for idx in train_indices:
        if idx < 0 or idx >= len(train_full):
            raise IndexError(f"Train index {idx} out of range [0, {len(train_full) - 1}]")
        print(f"\n{'='*60}\nTrain index {idx}\n{'='*60}")
        x_true, x_pred, diff, mse, mae = generate_and_plot_sample(
            train_full[idx], mean, std, method=method, title_prefix=f"Train[{idx}]"
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
            test_full[idx], mean, std, method=method, title_prefix=f"Test[{idx}]"
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

    samples_dir = "/Users/eugenekim/2dNS_Conditional_Diffusion/Interpolation/samples"
    os.makedirs(samples_dir, exist_ok=True)
    train_save = os.path.join(samples_dir, f"train_samples_3x3_interpolation_{method}.png")
    test_save = os.path.join(samples_dir, f"test_samples_3x3_interpolation_{method}.png")

    _plot_split_grid(train_rows, "train", train_save, method=method)
    _plot_split_grid(test_rows, "test", test_save, method=method)

    all_rows = train_rows + test_rows
    print(f"\n{'='*60}\nSummary ({len(all_rows)} samples)\n{'='*60}")
    for row in all_rows:
        print(
            f"  {row['split']} idx {row['index']:5d} | "
            f"MSE: {row['mse']:.6f} | MAE: {row['mae']:.6f}"
        )
    print(f"{'='*60}\n")


if __name__ == "__main__":
    data_path = "/Users/eugenekim/2dNS_Conditional_Diffusion/NSE_Data(Noisy).npy"
    compare_train_and_test_samples(
        data_path=data_path,
        method="linear",
        save_path=None,
        train_indices=[49999, 50000, 50001],
        test_indices=[0, 1, 2],
    )
