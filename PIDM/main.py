import os, yaml
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from tqdm import tqdm
from src.data_utils import *
from torch.utils.data import DataLoader
from src.denoising_utils import *
from src.cgf_model import Unet3D
# from src.residuals_darcy import ResidualsDarcy
# from src.residuals_mechanics_K import ResidualsMechanics
from src.residuals import Residuals
import numpy as np

name = 'run_1'
wandb_track = False # set to True to track training with wandb

load_model_flag = False # set to True to load a model
if load_model_flag:
    name = 'your_pretrained_model'
    load_path = './trained_models/' + name
    load_model_step = 0
    config = yaml.safe_load(Path(load_path, 'model', 'model.yaml').read_text())
else:
    config = yaml.safe_load(Path('/Users/eugenekim/2dNS_Conditional_Diffusion/PIDM/model.yaml').read_text())

# diffusion parameters
if config['x0_estimation'] == 'mean':
    use_ddim_x0 = False
elif config['x0_estimation'] == 'sample':
    use_ddim_x0 = True
ddim_steps = config['ddim_steps']
residual_grad_guidance = config['residual_grad_guidance'] # gradient guidance scale as in https://www.sciencedirect.com/science/article/pii/S0021999123000670
# residual corrections (can be changed after training since only affects inference) similar to https://arxiv.org/abs/2312.10527
correction_mode = config['correction_mode'] # 'x0', 'xt', CoCoGen use xt
M_correction = config['M_correction'] # correction steps after x0
N_correction = config['N_correction'] # correction steps before x0
fd_acc = config['fd_acc'] # finite difference accuracy
c_data = config['c_data']
c_residual = config['c_residual']
c_ineq = config['c_ineq']
lambda_opt = config['lambda_opt'] # (negative sign corresponds to max.)
diff_steps = config['diff_steps']
use_dynamic_threshold = False
self_condition = False

# evaluation params
test_eval_freq = 500
sample_freq = 2000
full_sample_freq = 10000
ema_start = 1000
ema = EMA(0.99)
topopt_eval = True # evaluate topopt metrics (only for mechanics as governing equations)
use_double = False
no_samples = 8
save_output = True
eval_residuals = True
create_gif = True  # Enable GIF generation of denoising process

# training parameters and datasets
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

data_paths = None
input_dim = 3 # (w, u, v)
output_dim = 3
pixels_at_boundary = True
domain_length = np.pi
reverse_d1 = True
bcs = 'periodic' # none or periodic
pixels_per_dim = 64
return_optimizer = False
return_inequality = False
train_batch_size = 32
train_iterations = 10000  # Total training iterations
if name == 'local_test':
    no_samples = 1
    train_batch_size = 8
    train_iterations = 1000
sigmoid_last_channel = False

# Load NS dataset
print('Loading dataset...')
h5_path = '/Users/eugenekim/2dNS_Conditional_Diffusion/NSE_Data(Noisy).npy'  # Dataset is in project root
# if not os.path.exists(h5_path):
#     h5_path = '../dns_dataset.h5'
if not os.path.exists(h5_path):
    h5_path = 'dns_dataset.h5'
full_dataset = Dataset(h5_path, use_double=use_double)
print(f'Total samples: {len(full_dataset)}')

# Split into train/validation (80/20 split)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
ds, ds_valid = torch.utils.data.random_split(full_dataset, [train_size, val_size])
print(f'Train samples: {len(ds)}, Validation samples: {len(ds_valid)}')

if use_double:
    torch.set_default_dtype(torch.float64)

dl = cycle(DataLoader(ds, batch_size = train_batch_size, shuffle=True))
dl_valid = cycle(DataLoader(ds_valid, batch_size = train_batch_size, shuffle=False))

# diffusion utils
diffusion_utils = DenoisingDiffusion(diff_steps, device, residual_grad_guidance)


model = Unet3D(dim = 32, channels = output_dim, sigmoid_last_channel = sigmoid_last_channel).to(device)
if load_model_flag:
    load_model(Path(load_path, 'model', 'checkpoint_' + str(load_model_step) + '.pt'), model)
ema.register(model)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Number of trainable parameters: {num_params}')

# residual computation for Navier-Stokes
residuals = Residuals(
    model=model, 
    pixels_per_dim=pixels_per_dim, 
    device=device, 
    domain_length=domain_length,
    visc=1e-3,  # Viscosity from your 2dNS.ipynb
    use_ddim_x0=use_ddim_x0, 
    ddim_steps=ddim_steps
)

optimizer = optim.Adam(model.parameters(), lr=1.e-4)

if wandb_track:
    import wandb
    wandb.init(project='pi_diffusion', name=name)
    log_fn = wandb.log
else:
    log_fn = noop
log_freq = 20
    
output_save_dir = f'./trained_models/{name}'
os.makedirs(output_save_dir, exist_ok=True)

# Save initial checkpoint at step 0 (before training)
save_model(config, model, 0, output_save_dir)
print("Saved initial checkpoint at step 0")

pbar = tqdm(range(train_iterations+1))
for iteration in pbar:
    model.train()
    cur_batch = next(dl).to(device)
    loss, data_loss, residual_loss, ineq_loss, opt_loss = diffusion_utils.model_estimation_loss(
                cur_batch, residual_func = residuals, c_data = c_data, c_residual = c_residual,
                c_ineq = c_ineq, lambda_opt = lambda_opt)    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()        
    # logging
    if iteration % log_freq == 0:
        pbar.set_description(f'training loss: {loss.item():.3e}')
        log_fn({'loss': loss.item()}, step=iteration)
        log_fn({'loss_data': data_loss}, step=iteration)
        log_fn({'residual_mean_abs': residual_loss}, step=iteration)
        if c_ineq > 0:
            log_fn({'loss_inequality': ineq_loss}, step=iteration)
        if lambda_opt > 0:
            log_fn({'loss_optimization': opt_loss}, step=iteration)
    # ema update
    if iteration > ema_start:
        ema.update(model)

    # evaluation on validation set
    model.eval()
    ema.ema(residuals.model)
    if iteration % test_eval_freq == 0 and exists(dl_valid):
        cur_test_batch = next(dl_valid).to(device)
        # NOTE: we do not use torch.no_grad() since we may require residual gradient for classifier-free guidance
        loss_test, data_loss_test, residual_loss_test, ineq_loss_test, opt_loss_test = diffusion_utils.model_estimation_loss(
                    cur_test_batch, residual_func = residuals, c_data = c_data, c_residual = c_residual,
                    c_ineq = c_ineq, lambda_opt = lambda_opt)
        
        print(f'test loss at iteration {iteration}: {loss_test:.3e}')
        log_fn({'loss_test': loss_test.item()}, step=iteration)
        log_fn({'loss_data_test': data_loss_test}, step=iteration)
        log_fn({'residual_mean_abs_test': residual_loss_test}, step=iteration)
        if c_ineq > 0:
            log_fn({'loss_inequality_test': ineq_loss_test}, step=iteration)
        if lambda_opt > 0:
            log_fn({'loss_optimization_test': opt_loss_test}, step=iteration)

    # generate and evaluate samples
    if (iteration % sample_freq == 0) or (iteration == train_iterations):        
        # if gov_eqs == 'darcy':
        #     conditioning_input = None
        #     sample_shape = (no_samples, output_dim, pixels_per_dim, pixels_per_dim)
        # elif gov_eqs == 'mechanics':
        #     cur_batch = next(dl_valid).to(device)
        #     if cur_batch.shape[0] < no_samples:
        #         no_samples = cur_batch.shape[0] # reduce no_samples to batch size
        #     sample_shape = (no_samples, output_dim, pixels_per_dim+1, pixels_per_dim+1)
        #     cur_batch = cur_batch[torch.randperm(cur_batch.shape[0], device = device)[:no_samples]]
        #     conditioning, x_0, bcs = torch.tensor_split(cur_batch, (3, 6), dim=1)
        #     conditioning_input = (conditioning, bcs, x_0)            
        #     # save conditioning data for later evaluation
        #     cond_data = torch.cat((conditioning, x_0, bcs), dim=1)
        #     for cur_sample in range(no_samples):
        #         for channel_idx in range(cond_data.shape[1]):
        #             os.makedirs(output_save_dir + f'/training/step_{iteration}/sample_{cur_sample}', exist_ok=True)
        #             np.savetxt(output_save_dir + f'/training/step_{iteration}/sample_{cur_sample}/cond_channel_{channel_idx}.csv', cond_data[cur_sample, channel_idx].detach().cpu().numpy(), delimiter=',')

        conditioning_input = None
        sample_shape = (no_samples, output_dim, pixels_per_dim, pixels_per_dim)
        
        output = diffusion_utils.p_sample_loop(conditioning_input, sample_shape, 
                                save_output=save_output, surpress_noise=True, 
                                use_dynamic_threshold=use_dynamic_threshold, 
                                residual_func=residuals, eval_residuals = eval_residuals, 
                                return_optimizer = return_optimizer, return_inequality = return_inequality,
                                M_correction = M_correction, N_correction = N_correction, correction_mode = correction_mode)
        
        if eval_residuals:
            seqs = output[0]
            residual = output[1]['residual']
            residual = residual.abs().mean(dim=tuple(range(1, residual.ndim))) # reduce to batch dim
            # Ensure residual is at least 1D (handles no_samples=1 case)
            if residual.ndim == 0:
                residual = residual.unsqueeze(0)
            if return_optimizer:
                optimized_quant = output[1]['optimized_quant']
            if return_inequality:
                ineq = output[1]['inequality_quant']
        else:
            seqs = output
            
        output_save_dir_step = output_save_dir + f'/training/step_{iteration}/'
        os.makedirs(output_save_dir_step, exist_ok=True)
                
        labels = ['sample', 'model_output']
        for seq_idx, seq in enumerate(seqs):

            # NOTE: We here only evaluate the sample at the final timestep and skip model_output as this is identical (since no noise is applied in last step).
            if seq_idx == 1:
                continue

            seq = torch.stack(seq, dim=0)

            if len(seq.shape) == 6:
                seq = seq.squeeze(-3)
                
            last_preds = seq[-1].numpy()
            sel_samples = np.arange(len(last_preds))
            channels = np.arange(output_dim)

            for sel_sample in sel_samples:
                for sel_channel in channels:
                    last_pred = last_preds[sel_sample, sel_channel]
                    last_pred_normalized = (last_pred - last_pred.min()) / (last_pred.max() - last_pred.min()) # normalize to [0,1]

                    image = np.uint8(last_pred_normalized * 255)
                    fig, ax = plt.subplots()
                    ax.imshow(image, cmap='gray', vmin=0, vmax=255)
                    ax.axis('off')
                    if eval_residuals:
                        # Safe indexing: handle case where residual has fewer elements
                        res_idx = min(sel_sample, len(residual) - 1)
                        title = f'eq: {residual[res_idx]:.2e}'
                        if return_optimizer:
                            opt_idx = min(sel_sample, len(optimized_quant) - 1)
                            title += f'\nopt: {optimized_quant[opt_idx]:.2f}'
                        if return_inequality:
                            ineq_idx = min(sel_sample, len(ineq) - 1)
                            title += f'\nineq: {ineq[ineq_idx]:.2e}'
                        plt.title(title, color='green')
                    filename = labels[seq_idx] + '_sample_' + str(sel_sample) + '_' + str(sel_channel) + '.png'
                    plt.savefig(output_save_dir_step + filename, bbox_inches='tight', pad_inches=0)
                    plt.close(fig)

                    os.makedirs(output_save_dir_step + f'/sample_{sel_sample}/', exist_ok=True)
                    np.savetxt(output_save_dir_step + f'/sample_{sel_sample}/' + labels[seq_idx] + '_' + str(sel_channel) + '.csv', last_pred, delimiter=',')

                    if create_gif:
                        sel_seq = seq[:, sel_sample, sel_channel].detach().cpu().numpy()
                        image_array_to_gif(sel_seq, output_save_dir_step + f'/sample_{sel_sample}/' + labels[seq_idx] + '_' + str(sel_channel) + '.gif')


        if eval_residuals:
            residuals_array = residual.detach().cpu().numpy()
            ineq_array = ineq.detach().cpu().numpy() if return_inequality else None
            optimized_quant_array = optimized_quant.detach().cpu().numpy() if return_optimizer else None

            # logging
            log_fn({'residual_mean_abs_samples': np.nanmean(residuals_array)}, step=iteration)
            log_fn({'residual_median_abs_samples': np.nanmedian(residuals_array)}, step=iteration)
            
            # Create DataFrame with matching array lengths
            num_residuals = len(residuals_array)
            df_data = {
                'Sample Index': list(range(num_residuals)) + ['Mean'],
                'Residuals (abs)': list(residuals_array) + [np.nanmean(residuals_array)]
            }
            if return_inequality:
                df_data['Inequality'] = list(ineq_array) + [np.nanmean(ineq_array)]
            if return_optimizer:
                df_data['Optimized quantity'] = list(optimized_quant_array) + [np.nanmean(optimized_quant_array)]
            
            df = pd.DataFrame(df_data)
            csv_path = os.path.join(output_save_dir_step, 'sample_statistics.csv')
            df.to_csv(csv_path, index=False)

        if iteration > 0:
            save_model(config, model, iteration, output_save_dir)

    ema.restore(residuals.model)

if wandb_track:
    wandb.finish()