import torch
import numpy as np
from src.data_utils import generalized_image_to_b_xy_c, generalized_b_xy_c_to_image
try:
    # PyTorch 2.0+
    from torch.func import vmap, jacfwd
except ImportError:
    # Older PyTorch with functorch
    from functorch import vmap, jacfwd

class Residuals:
    def __init__(self, model, pixels_per_dim, device='cpu', domain_length=np.pi, visc=1e-3, use_ddim_x0=False, ddim_steps=0):
        """
        Initialize the residual evaluation.

        :param model: The neural network model to compute the residuals for.
        :param n_steps: Number of steps for time discretization.
        :param E: Young's Modulus.
        :param nu: Poisson's Ratio.
        """
        self.model = model
        self.pixels_per_dim = pixels_per_dim
        self.device = device
        self.domain_length = domain_length
        self.visc = visc
        # Setup Fourier space grid for spectral derivatives
        N = pixels_per_dim
        k = (2 * np.pi / domain_length) * torch.fft.fftfreq(N, device=device)
        kx = k.view(-1, 1).repeat(1, N)
        ky = k.view(1, -1).repeat(N, 1)
        self.kx = kx
        self.ky = ky
        self.k_squared = kx**2 + ky**2
        
        # Avoid division by zero for k=0 mode
        self.k_squared_safe = self.k_squared.clone()
        self.k_squared_safe[0, 0] = 1.0

        self.use_ddim_x0 = use_ddim_x0
        self.ddim_steps = ddim_steps

    def compute_residual(self, input, reduce='none', return_model_out=False, 
                        pass_through=False, ddim_func=None, sample=False, **kwargs):
        """
        Compute Navier-Stokes physics residuals.
        """
        
        # --- Step 1: Get model output (denoised prediction) ---
        if pass_through:
            x0_pred = input  # Direct input
            model_out = x0_pred
        else:
            noisy_in, time = input[0]
            if self.use_ddim_x0:
                x0_pred, model_out = ddim_func(noisy_in, time, self.model, 
                                               noisy_in.shape, self.ddim_steps, 0.)
            else:
                x0_pred = self.model(noisy_in, time)
                model_out = x0_pred
        
        # x0_pred shape: (batch, 3, 64, 64) = (batch, channels, H, W)
        # channels: [w, u, v] = [vorticity, u-velocity, v-velocity]
        batch_size = x0_pred.shape[0]
        
        # --- Step 2: Extract fields ---
        w = x0_pred[:, 0:1]  # vorticity: (batch, 1, H, W)
        u = x0_pred[:, 1:2]  # u-velocity
        v = x0_pred[:, 2:3]  # v-velocity
        
        # ============================================================
        # === PHYSICS CONSTRAINTS: NAVIER-STOKES EQUATIONS ===
        # ============================================================
        
        # Direct physics residuals (no streamfunction needed):
        # Residual 1: ω - (∂v/∂x - ∂u/∂y) = 0  (vorticity definition)
        # Residual 2: ∂u/∂x + ∂v/∂y = 0  (incompressibility)
        #
        # Note: Cannot use vorticity transport equation (∂ω/∂t + u·∇ω = ν∇²ω + f)
        #       because dataset treats each timestep independently (no temporal pairs)

        # --- Compute spatial derivatives in Fourier space ---
        
        # Velocity derivatives
        u_fft = torch.fft.rfft2(u.squeeze(1))
        v_fft = torch.fft.rfft2(v.squeeze(1))
        
        u_x = torch.fft.irfft2(1j * self.kx[:, :u_fft.shape[-1]] * u_fft,
                               s=(self.pixels_per_dim, self.pixels_per_dim))
        u_y = torch.fft.irfft2(1j * self.ky[:, :u_fft.shape[-1]] * u_fft,
                               s=(self.pixels_per_dim, self.pixels_per_dim))
        v_x = torch.fft.irfft2(1j * self.kx[:, :v_fft.shape[-1]] * v_fft,
                               s=(self.pixels_per_dim, self.pixels_per_dim))
        v_y = torch.fft.irfft2(1j * self.ky[:, :v_fft.shape[-1]] * v_fft,
                               s=(self.pixels_per_dim, self.pixels_per_dim))
        
        # --- Physics Residuals ---
        
        # Residual 1: Vorticity-velocity relationship
        # ω = ∂v/∂x - ∂u/∂y (definition of vorticity from velocity field)
        vorticity_from_velocity = v_x - u_y
        residual_vorticity = w.squeeze(1) - vorticity_from_velocity  # Should be ~0
        
        # Residual 2: Incompressibility constraint
        # ∇·u = ∂u/∂x + ∂v/∂y = 0
        divergence = u_x + v_y  # Should be ~0
        
        # --- Combine residuals ---
        # Convert to (batch, pixels*pixels, num_residuals) format
        residual_vorticity = generalized_image_to_b_xy_c(residual_vorticity.unsqueeze(1))
        divergence = generalized_image_to_b_xy_c(divergence.unsqueeze(1))
        
        # Stack all residuals
        residual = torch.cat([residual_vorticity, divergence], dim=-1)
        # Shape: (batch, 64*64, 2)
        
        # ============================================================
        # === END PHYSICS CONSTRAINTS ===
        # ============================================================
        
        output = {}
        output['residual'] = residual
        
        if return_model_out:
            output['model_out'] = model_out
        
        # Reduce based on mode
        if reduce == 'full':
            return {k: v.mean() for k, v in output.items()}
        elif reduce == 'per-batch':
            return {k: v.mean(dim=tuple(range(1, v.ndim))) if v.ndim > 1 and k != 'model_out' else v 
                    for k, v in output.items()}
        elif reduce == 'none':                 
            return output
        else:
            raise ValueError('Unknown reduction method.')
        
    def residual_correction(self, x0_pred_in):
        """
        CoCoGen-style residual correction for Navier-Stokes.
        Corrects all three fields: vorticity, u-velocity, v-velocity
        """
        # Ensure the model output is in the correct shape
        assert len(x0_pred_in.shape) == 3, 'Model output must be a tensor shaped as b_xy_c.'

        x0_pred = x0_pred_in.detach().clone()
        x0_pred.requires_grad_(True)

        # Compute residuals
        residual_x0_pred = self.compute_residual(generalized_b_xy_c_to_image(x0_pred), 
                                                 pass_through=True)['residual']
        
        # Gradient of squared residual w.r.t. ALL channels (not just channel 0)
        residual_loss = torch.sum(residual_x0_pred**2)
        dr_dx = torch.autograd.grad(residual_loss, x0_pred)[0]  # (batch, 64*64, 3)
        
        # Compute Jacobian for step size estimation
        jacobian_batch_size = 1  # Reduced batch size to avoid OOM
        jacobian_vmap = vmap(jacfwd(self.compute_residual_direct, argnums=0, has_aux=False), 
                             in_dims=0, out_dims=0)
        num_batches = x0_pred.shape[0] // jacobian_batch_size + \
                      (0 if x0_pred.shape[0] % jacobian_batch_size == 0 else 1)
        jacobian_max_values = []
        
        for i in range(num_batches):
            batch_inputs = x0_pred[i*jacobian_batch_size:(i+1)*jacobian_batch_size]
            # jacobian shape: (batch, 64*64, 4_residuals, 64*64, 3_channels)
            jacobian_vmapped = jacobian_vmap(batch_inputs)
            
            # Get max value across all residuals and channels
            batch_max_values = torch.max(jacobian_vmapped.abs().reshape(len(batch_inputs), -1), 
                                         dim=1)[0]
            jacobian_max_values.extend(batch_max_values.tolist())
            del jacobian_vmapped, batch_max_values
            torch.cuda.empty_cache()

        max_dr = torch.tensor(jacobian_max_values).to(x0_pred.device)
        max_dr = torch.clamp(max_dr, max=1e12)
        correction_eps = 1.e-6 / max_dr

        # Apply correction to ALL channels (not just channel 0!)
        # correction_eps shape: (batch,)
        # dr_dx shape: (batch, 64*64, 3)
        x0_pred_in -= correction_eps.unsqueeze(1).unsqueeze(2) * dr_dx.detach()

        # Compute residual again after correction
        residual_corrected = self.compute_residual(generalized_b_xy_c_to_image(x0_pred_in), 
                                                   pass_through=True)['residual']
        return x0_pred_in, residual_corrected
        
    # Compute the residual directly to simplify jacfwd call
    def compute_residual_direct(self, x0_output):
        """
        Simplified residual computation for Jacobian calculations.
        Used by residual_correction for CoCoGen-style corrections.
        
        Input: x0_output in b_xy_c format (batch, 64*64, 3)
        Output: residual in b_xy_c format (batch, 64*64, 2)
        
        Residuals:
        1. Vorticity-velocity: ω - (∂v/∂x - ∂u/∂y) = 0
        2. Incompressibility: ∂u/∂x + ∂v/∂y = 0
        """
        # Ensure the model output is in the correct shape
        if x0_output.ndim == 2:
            x0_output = x0_output.unsqueeze(0)
        assert len(x0_output.shape) == 3, 'Model output must be a tensor shaped as b_xy_c.'
        
        # Convert to image format (batch, 3, 64, 64)
        x0_output = generalized_b_xy_c_to_image(x0_output)
        
        # Extract fields
        w = x0_output[:, 0:1]  # vorticity
        u = x0_output[:, 1:2]  # u-velocity
        v = x0_output[:, 2:3]  # v-velocity
        
        # === Compute Spatial Derivatives in Fourier Space ===
        
        # Velocity FFTs
        u_fft = torch.fft.rfft2(u.squeeze(1))
        v_fft = torch.fft.rfft2(v.squeeze(1))
        
        # Velocity derivatives
        u_x = torch.fft.irfft2(1j * self.kx[:, :u_fft.shape[-1]] * u_fft,
                               s=(self.pixels_per_dim, self.pixels_per_dim))
        u_y = torch.fft.irfft2(1j * self.ky[:, :u_fft.shape[-1]] * u_fft,
                               s=(self.pixels_per_dim, self.pixels_per_dim))
        v_x = torch.fft.irfft2(1j * self.kx[:, :v_fft.shape[-1]] * v_fft,
                               s=(self.pixels_per_dim, self.pixels_per_dim))
        v_y = torch.fft.irfft2(1j * self.ky[:, :v_fft.shape[-1]] * v_fft,
                               s=(self.pixels_per_dim, self.pixels_per_dim))
        
        # Residual 1: Vorticity-velocity relationship
        # ω = ∂v/∂x - ∂u/∂y
        vorticity_from_velocity = v_x - u_y
        residual_vorticity = w.squeeze(1) - vorticity_from_velocity
        
        # Residual 2: Incompressibility
        # ∇·u = ∂u/∂x + ∂v/∂y = 0
        divergence = u_x + v_y
        
        # Convert to b_xy_c format
        residual_vorticity = generalized_image_to_b_xy_c(residual_vorticity.unsqueeze(1))
        divergence = generalized_image_to_b_xy_c(divergence.unsqueeze(1))
        
        # Concatenate all residuals (batch, 64*64, 2)
        residual = torch.cat([residual_vorticity, divergence], dim=-1)
        
        return residual 