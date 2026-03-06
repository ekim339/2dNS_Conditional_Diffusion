import numpy as np
import torch
import matplotlib.pyplot as plt
from ConditionalDiffusion import (
    ConditionalDDPM, DDPMTrainer, DiffusionConfig,
    default_device
)

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
    Generate a single sample from a true field and return original, generated, and difference.
    
    Returns:
        x_true: Original field (numpy array)
        x_pred: Generated field (numpy array)
        diff: Difference (numpy array)
        mse: MSE value
        mae: MAE value
    """
    # Ensure x0_true is 2D (64, 64)
    if x0_true.dim() > 2:
        x0_true = x0_true.squeeze()
    assert x0_true.shape == (64, 64), f"Expected (64, 64), got {x0_true.shape}"
    
    # Normalize
    x0_true_norm = (x0_true - mean) / (std + 1e-8)
    
    # Build sparse observation y (8x8) - match dataset exactly
    coords = torch.arange(0, 64, 8, dtype=torch.long)
    c = coords
    y_sparse = x0_true_norm[c][:, c]  # (8, 8) - same as dataset
    
    # Verify sparse observation shape and values
    assert y_sparse.shape == (8, 8), f"Expected y_sparse shape (8, 8), got {y_sparse.shape}"
    
    # Prepare for model input
    y_input = y_sparse.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 8, 8)
    assert y_input.shape == (1, 1, 8, 8), f"Expected y_input shape (1, 1, 8, 8), got {y_input.shape}"
    
    # Generate single sample
    print("Generating sample...", end=" ", flush=True)
    
    with torch.no_grad():
        x_pred_norm = trainer.sample_cfg(
            y=y_input,
            guidance_scale=guidance_scale,
            shape=(1, 1, 64, 64),
        )
    
    # Unnormalize
    if x_pred_norm.dim() == 4:
        x_pred_norm = x_pred_norm.squeeze()
    x_pred = x_pred_norm.cpu() * (std + 1e-8) + mean  # (64, 64)
    
    print("Done.")
    
    x_true = x0_true.cpu()  # (64, 64)
    
    # Convert to numpy
    x_true = x_true.numpy()
    x_pred = x_pred.numpy()
    diff = x_true - x_pred
    
    # Compute metrics
    mse = np.mean((x_true - x_pred) ** 2)
    mae = np.mean(np.abs(x_true - x_pred))
    
    return x_true, x_pred, diff, mse, mae


def compare_train_and_test_samples(
    ckpt_path: str,
    data_path: str,
    guidance_scale: float = None,  # None = use checkpoint default
    save_path: str = None,
):
    """
    Generate single samples from first train sample and first test sample, then plot comparisons.
    """
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
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
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
    
    cfg = DiffusionConfig(**cfg_dict)
    trainer = DDPMTrainer(model, cfg, device)
    
    # Get first train sample (index 0)
    print(f"\n{'='*60}")
    print(f"Processing Train Sample (index 0)")
    print(f"{'='*60}")
    train_sample = train_full[50000]
    x_true_train, x_pred_train, diff_train, mse_train, mae_train = generate_and_plot_sample(
        trainer, train_sample, mean, std, device, guidance_scale, "Train"
    )
    print(f"Train Sample MSE: {mse_train:.6f}, MAE: {mae_train:.6f}")
    
    # Get first test sample (index 0 in test set)
    print(f"\n{'='*60}")
    print(f"Processing Test Sample (index 0 in test set)")
    print(f"{'='*60}")
    test_sample = test_full[0]
    x_true_test, x_pred_test, diff_test, mse_test, mae_test = generate_and_plot_sample(
        trainer, test_sample, mean, std, device, guidance_scale, "Test"
    )
    print(f"Test Sample MSE: {mse_test:.6f}, MAE: {mae_test:.6f}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: Train sample
    # Original train sample
    im1 = axes[0, 0].imshow(x_true_train, cmap='viridis', aspect='auto')
    axes[0, 0].set_title('Train Sample: Original', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    for i in range(1, 8):
        axes[0, 0].axvline(x=i * 8 - 0.5, color='white', linewidth=1, alpha=0.6, linestyle='--')
        axes[0, 0].axhline(y=i * 8 - 0.5, color='white', linewidth=1, alpha=0.6, linestyle='--')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
    
    # Generated train sample
    im2 = axes[0, 1].imshow(x_pred_train, cmap='viridis', aspect='auto')
    axes[0, 1].set_title(f'Train Sample: Generated\nMSE: {mse_train:.6f}', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    for i in range(1, 8):
        axes[0, 1].axvline(x=i * 8 - 0.5, color='white', linewidth=1, alpha=0.6, linestyle='--')
        axes[0, 1].axhline(y=i * 8 - 0.5, color='white', linewidth=1, alpha=0.6, linestyle='--')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
    
    # Difference train sample
    im3 = axes[0, 2].imshow(diff_train, cmap='RdBu_r', aspect='auto',
                           vmin=-np.abs(diff_train).max(), vmax=np.abs(diff_train).max())
    axes[0, 2].set_title(f'Train Sample: Difference\nMAE: {mae_train:.6f}', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    for i in range(1, 8):
        axes[0, 2].axvline(x=i * 8 - 0.5, color='white', linewidth=1, alpha=0.6, linestyle='--')
        axes[0, 2].axhline(y=i * 8 - 0.5, color='white', linewidth=1, alpha=0.6, linestyle='--')
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046)
    
    # Row 2: Test sample
    # Original test sample
    im4 = axes[1, 0].imshow(x_true_test, cmap='viridis', aspect='auto')
    axes[1, 0].set_title('Test Sample: Original', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    for i in range(1, 8):
        axes[1, 0].axvline(x=i * 8 - 0.5, color='white', linewidth=1, alpha=0.6, linestyle='--')
        axes[1, 0].axhline(y=i * 8 - 0.5, color='white', linewidth=1, alpha=0.6, linestyle='--')
    plt.colorbar(im4, ax=axes[1, 0], fraction=0.046)
    
    # Generated test sample
    im5 = axes[1, 1].imshow(x_pred_test, cmap='viridis', aspect='auto')
    axes[1, 1].set_title(f'Test Sample: Generated\nMSE: {mse_test:.6f}', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    for i in range(1, 8):
        axes[1, 1].axvline(x=i * 8 - 0.5, color='white', linewidth=1, alpha=0.6, linestyle='--')
        axes[1, 1].axhline(y=i * 8 - 0.5, color='white', linewidth=1, alpha=0.6, linestyle='--')
    plt.colorbar(im5, ax=axes[1, 1], fraction=0.046)
    
    # Difference test sample
    im6 = axes[1, 2].imshow(diff_test, cmap='RdBu_r', aspect='auto',
                           vmin=-np.abs(diff_test).max(), vmax=np.abs(diff_test).max())
    axes[1, 2].set_title(f'Test Sample: Difference\nMAE: {mae_test:.6f}', fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')
    for i in range(1, 8):
        axes[1, 2].axvline(x=i * 8 - 0.5, color='white', linewidth=1, alpha=0.6, linestyle='--')
        axes[1, 2].axhline(y=i * 8 - 0.5, color='white', linewidth=1, alpha=0.6, linestyle='--')
    plt.colorbar(im6, ax=axes[1, 2], fraction=0.046)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    else:
        plt.show()
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"Train Sample (index 0):")
    print(f"  MSE: {mse_train:.6f}")
    print(f"  MAE: {mae_train:.6f}")
    print(f"\nTest Sample (index 0 in test set):")
    print(f"  MSE: {mse_test:.6f}")
    print(f"  MAE: {mae_test:.6f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Configuration
    ckpt_path = "/Users/eugenekim/2dNS_Conditional_Diffusion/checkpoint/w=0.3p=0.1.pt"
    data_path = "/Users/eugenekim/2dNS_Conditional_Diffusion/NSE_Data(Noisy).npy"
    guidance_scale = None  # None = use checkpoint's guidance_scale, or set explicitly (e.g., 4.0)
    
    compare_train_and_test_samples(
        ckpt_path=ckpt_path,
        data_path=data_path,
        guidance_scale=guidance_scale,
        save_path=None,  # Set to a path like "comparison.png" to save instead of displaying
    )

