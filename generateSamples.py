import numpy as np
import torch
import matplotlib.pyplot as plt
from ConditionalDiffusion import (
    ConditionalDDPM, DDPMTrainer, DiffusionConfig,
    default_device
)

def generate_multiple_samples_and_average(
    ckpt_path: str,
    data_path: str,
    test_index: int = 10,
    num_simulations: int = 10,
    guidance_scale: float = 4.0,
    save_path: str = None,
):
    """
    Generate multiple samples from the same sparse input and average them pixel-wise.
    
    Args:
        ckpt_path: Path to checkpoint file (.pt)
        data_path: Path to data file (.npy)
        test_index: Index of test sample to use
        num_simulations: Number of simulations to generate and average
        guidance_scale: CFG guidance scale for sampling
        save_path: Optional path to save the figure (if None, displays interactively)
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
    test_full = data_t[n_train:].float()
    
    print(f"Total samples: {N}")
    print(f"Train samples: {n_train} (80%)")
    print(f"Test samples: {len(test_full)} (20%)")
    
    # Select test index
    if test_index is None:
        test_index = np.random.randint(0, len(test_full))
        print(f"Randomly selected test index: {test_index}")
    else:
        if test_index < 0 or test_index >= len(test_full):
            raise ValueError(f"test_index must be between 0 and {len(test_full)-1}, got {test_index}")
        print(f"Using test index: {test_index}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    mean = float(ckpt["train_mean"])
    std = float(ckpt["train_std"])
    cfg_dict = ckpt["cfg"]
    
    print(f"Train mean: {mean:.6f}")
    print(f"Train std: {std:.6f}")
    
    # Rebuild model
    model = ConditionalDDPM(T=cfg_dict["T"], emb_dim=256, base_ch=64).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    
    cfg = DiffusionConfig(**cfg_dict)
    trainer = DDPMTrainer(model, cfg, device)
    
    # Get the true field
    x0_true = test_full[test_index]  # (64, 64)
    x0_true_norm = (x0_true - mean) / (std + 1e-8)
    
    # Build sparse observation y (8x8) - must match dataset extraction exactly
    coords = torch.arange(0, 64, 8, dtype=torch.long)
    y_sparse = x0_true_norm[coords][:, coords]  # (8, 8)
    
    # Prepare for model input
    y_input = y_sparse.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 8, 8)
    
    print(f"\nGenerating {num_simulations} simulations with guidance_scale={guidance_scale}...")
    print("This may take a moment (DDPM sampling is iterative)...")
    
    # Generate multiple samples
    all_samples = []
    with torch.no_grad():
        for sim_idx in range(num_simulations):
            print(f"  Simulation {sim_idx + 1}/{num_simulations}...", end=" ", flush=True)
            x_pred_norm = trainer.sample_cfg(
                y=y_input,
                guidance_scale=guidance_scale,
                shape=(1, 1, 64, 64),
            )
            # Unnormalize
            if x_pred_norm.dim() == 4:
                x_pred_norm = x_pred_norm.squeeze()
            x_pred = x_pred_norm.cpu() * (std + 1e-8) + mean  # (64, 64)
            all_samples.append(x_pred.numpy())
            print("✓")
    
    # Stack all samples and compute average
    all_samples_array = np.stack(all_samples, axis=0)  # (num_simulations, 64, 64)
    x_pred_avg = np.mean(all_samples_array, axis=0)  # (64, 64)
    x_pred_std = np.std(all_samples_array, axis=0)  # (64, 64) - standard deviation across simulations
    
    # Get true field
    x_true = x0_true.cpu().numpy()  # (64, 64)
    y_sparse_unnorm = y_sparse.cpu().numpy() * (std + 1e-8) + mean  # (8, 8)
    
    # Compute metrics for averaged prediction
    mse_avg = np.mean((x_true - x_pred_avg) ** 2)
    mae_avg = np.mean(np.abs(x_true - x_pred_avg))
    rmse_avg = np.sqrt(mse_avg)
    
    # Compute metrics for individual simulations
    mse_individual = [np.mean((x_true - sample) ** 2) for sample in all_samples]
    mae_individual = [np.mean(np.abs(x_true - sample)) for sample in all_samples]
    
    # Check sensor location matching for average
    coords_np = np.arange(0, 64, 8)
    x_pred_avg_at_sensors = x_pred_avg[coords_np][:, coords_np]
    sensor_mse_avg = np.mean((y_sparse_unnorm - x_pred_avg_at_sensors) ** 2)
    
    print(f"\n{'='*60}")
    print(f"Results for Test Sample {test_index}")
    print(f"{'='*60}")
    print(f"\nAveraged Prediction Metrics:")
    print(f"  MSE:  {mse_avg:.6f}")
    print(f"  MAE:  {mae_avg:.6f}")
    print(f"  RMSE: {rmse_avg:.6f}")
    print(f"  Sensor location MSE: {sensor_mse_avg:.6f}")
    
    print(f"\nIndividual Simulation Metrics:")
    print(f"  MSE range: [{np.min(mse_individual):.6f}, {np.max(mse_individual):.6f}]")
    print(f"  MSE mean: {np.mean(mse_individual):.6f} ± {np.std(mse_individual):.6f}")
    print(f"  MAE range: [{np.min(mae_individual):.6f}, {np.max(mae_individual):.6f}]")
    print(f"  MAE mean: {np.mean(mae_individual):.6f} ± {np.std(mae_individual):.6f}")
    
    print(f"\nImprovement from averaging:")
    print(f"  MSE improvement: {np.mean(mse_individual) - mse_avg:.6f}")
    print(f"  MAE improvement: {np.mean(mae_individual) - mae_avg:.6f}")
    print(f"{'='*60}\n")
    
    # Create visualization
    fig = plt.figure(figsize=(20, 14))
    
    # Row 1: True field, Averaged prediction, Error
    ax1 = plt.subplot(3, 3, 1)
    im1 = ax1.imshow(x_true, cmap='viridis', aspect='auto')
    ax1.set_title('True Field (64×64)', fontsize=14, fontweight='bold')
    ax1.axis('off')
    for i in range(1, 8):
        ax1.axvline(x=i * 8 - 0.5, color='white', linewidth=1, alpha=0.6, linestyle='--')
        ax1.axhline(y=i * 8 - 0.5, color='white', linewidth=1, alpha=0.6, linestyle='--')
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    
    ax2 = plt.subplot(3, 3, 2)
    im2 = ax2.imshow(x_pred_avg, cmap='viridis', aspect='auto')
    ax2.set_title(f'Averaged Prediction ({num_simulations} sims)', fontsize=14, fontweight='bold')
    ax2.axis('off')
    for i in range(1, 8):
        ax2.axvline(x=i * 8 - 0.5, color='white', linewidth=1, alpha=0.6, linestyle='--')
        ax2.axhline(y=i * 8 - 0.5, color='white', linewidth=1, alpha=0.6, linestyle='--')
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    ax3 = plt.subplot(3, 3, 3)
    error_avg = np.abs(x_true - x_pred_avg)
    im3 = ax3.imshow(error_avg, cmap='hot', aspect='auto')
    ax3.set_title(f'Absolute Error (Avg)\nMSE: {mse_avg:.6f}', fontsize=14, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046)
    
    # Row 2: Sparse input, Standard deviation, Difference
    ax4 = plt.subplot(3, 3, 4)
    y_upsampled = np.repeat(np.repeat(y_sparse_unnorm, 8, axis=0), 8, axis=1)
    im4 = ax4.imshow(y_upsampled, cmap='viridis', aspect='auto', interpolation='nearest')
    ax4.set_title('Sparse Input (8×8, upsampled)', fontsize=14, fontweight='bold')
    ax4.axis('off')
    for i in range(1, 8):
        ax4.axvline(x=i * 8 - 0.5, color='white', linewidth=1.5, alpha=0.8)
        ax4.axhline(y=i * 8 - 0.5, color='white', linewidth=1.5, alpha=0.8)
    plt.colorbar(im4, ax=ax4, fraction=0.046)
    
    ax5 = plt.subplot(3, 3, 5)
    im5 = ax5.imshow(x_pred_std, cmap='hot', aspect='auto')
    ax5.set_title(f'Std Dev Across Simulations', fontsize=14, fontweight='bold')
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046)
    
    ax6 = plt.subplot(3, 3, 6)
    diff_avg = x_true - x_pred_avg
    im6 = ax6.imshow(diff_avg, cmap='RdBu_r', aspect='auto', 
                     vmin=-np.abs(diff_avg).max(), vmax=np.abs(diff_avg).max())
    ax6.set_title('Difference (True - Avg)', fontsize=14, fontweight='bold')
    ax6.axis('off')
    plt.colorbar(im6, ax=ax6, fraction=0.046)
    
    # Row 3: Show a few individual simulations
    for idx in range(min(3, num_simulations)):
        ax = plt.subplot(3, 3, 7 + idx)
        im = ax.imshow(all_samples[idx], cmap='viridis', aspect='auto')
        ax.set_title(f'Simulation {idx + 1}\nMSE: {mse_individual[idx]:.6f}', 
                    fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    import sys
    
    # Configuration
    ckpt_path = "/Users/eugenekim/2dNS_Conditional_Diffusion/checkpoint/last.pt"
    data_path = "/Users/eugenekim/2dNS_Conditional_Diffusion/NSE_Data(Noisy).npy"
    guidance_scale = 4.0  # CFG guidance scale
    num_simulations = 10  # Number of simulations to generate and average
    
    # Allow command-line argument for test_index
    if len(sys.argv) > 1:
        try:
            test_index = int(sys.argv[1])
        except ValueError:
            print(f"Error: '{sys.argv[1]}' is not a valid integer index")
            print("Usage: python generateSamples.py [test_index]")
            sys.exit(1)
    else:
        test_index = 10561  # Default to the requested sample
    
    # Load data to check valid range
    data = np.load(data_path)
    data_t = torch.from_numpy(data)
    N = data_t.shape[0]
    n_train = int(0.8 * N)
    n_test = N - n_train
    
    # Check if this is a full dataset index or test set index
    original_index = test_index
    if test_index >= n_train:
        # This is a full dataset index, convert to test set index
        full_dataset_index = test_index
        test_index = test_index - n_train
        print(f"Full dataset index: {full_dataset_index} -> Test set index: {test_index}")
    elif test_index < 0:
        # Handle negative indices (relative to test set)
        test_index = n_test + test_index
        print(f"Using test set index: {test_index} (from negative index)")
    
    if test_index < 0 or test_index >= n_test:
        print(f"Error: test_index must be:")
        print(f"  - A test set index between 0 and {n_test-1}, OR")
        print(f"  - A full dataset index between {n_train} and {N-1}")
        print(f"  Got: {original_index}")
        sys.exit(1)
    
    if original_index >= n_train:
        print(f"Will visualize full dataset index: {original_index} (test set index: {test_index})")
    else:
        print(f"Will visualize test set index: {test_index}")
    print(f"  (Test set has {n_test} samples, test indices 0-{n_test-1})")
    print(f"  (Full dataset indices for test: {n_train} to {N-1})")
    
    # Generate multiple samples and average
    generate_multiple_samples_and_average(
        ckpt_path=ckpt_path,
        data_path=data_path,
        test_index=test_index,
        num_simulations=num_simulations,
        guidance_scale=guidance_scale,
        save_path=None,  # Set to a path like "averaged_sample.png" to save instead of displaying
    )
