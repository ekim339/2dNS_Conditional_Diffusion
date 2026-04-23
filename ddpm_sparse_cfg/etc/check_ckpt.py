import torch

ckpt_path = "checkpoint/w=10p=0.1.pt"
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

cfg = ckpt.get('cfg', {})
print("Checkpoint configuration:")
print(f"  Guidance scale: {cfg.get('guidance_scale', 'Not found')}")
print(f"  Drop prob: {cfg.get('drop_prob', 'Not found')}")
print(f"  T (timesteps): {cfg.get('T', 'Not found')}")
print(f"  Batch size: {cfg.get('batch_size', 'Not found')}")
print(f"  Epochs: {cfg.get('epochs', 'Not found')}")

