import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from cfgConditional import run_training


# NSE_data = np.load("/Users/eugenekim/2dNS_Conditional_Diffusion/NSE_Data(Noisy).npy")

# print(NSE_data.shape)


if __name__ == "__main__":

    # example: load your dataset
    # replace this with your actual loading
    NSE_data = np.load("/Users/eugenekim/2dNS_Conditional_Diffusion/NSE_Data(Noisy).npy")
    print(NSE_data.shape)

    ckpt_path, stats, cfg = run_training(NSE_data)


# import matplotlib.pyplot as plt
# import seaborn as sns

# sns.heatmap(NSE_data[90000, ::8, ::8])


# plt.show()



# condtitional diffusion model  (DDPm)


# Friday: 


# ((8, 8) iamge 


# true (64, 64) field


# CDM sample of (64, 64)


# CDM mean of (64, 64)
# CDM sample of (64, 64)
# CDM sample of (64, 64)
# CDM sample of (64, 64)
# CDM sample of (64, 64)
# CDM sample of (64, 64)
# CDM sample of (64, 64)
# CDM sample of (64, 64)
# CDM sample of (64, 64)
# CDM sample of (64, 64)



# torch(10, 64, 64).mean(dim=0)


# sample1: (0,0) -10 (0,1) 0
# sample2: (0,0) 10 (0,1) 0
# avg: (0, 0) 0 (0, 1)










(32, 32)
supershot

(1000, 10000)