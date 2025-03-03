import argparse
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

import model.inference as inference
from model.transformer_timeseries import TimeSeriesTransformer
from psd_torch import power_spectrum_error

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initial condition
test_data_path = "data/lorenz96_test.npy"
test_data = torch.tensor(np.load(test_data_path), device=device)
gen_data = torch.tensor(np.load("96/trajectory.npy"), device=device).squeeze(0)

# x = trajectory_np[:, 0]
# y = trajectory_np[:, 1]
# z = trajectory_np[:, 2]

x_hat = test_data[:, 0].cpu().numpy()
y_hat = test_data[:, 1].cpu().numpy()
z_hat = test_data[:, 2].cpu().numpy()

print(gen_data.shape, test_data.shape)
psd, spectrum_true, spectrum_gen = power_spectrum_error(gen_data.unsqueeze(0), test_data[15:, :].unsqueeze(0))

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# # ax.plot(x, y, z)
# ax.plot(x_hat, y_hat, z_hat, linewidth=0.1)

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.title('True 3D Trajectory')
# plt.show()

# Plot and save the PSD
ax1 = plt.subplot(111)
ax1.plot(spectrum_true.squeeze().cpu().detach().numpy(), label="True")
ax1.plot(spectrum_gen.squeeze().cpu().detach().numpy(), label="Generated")
ax1.set_title("Power Spectrum Density")
ax1.set_yscale("log")
ax1.set_xscale("log")
ax1.legend()

plt.savefig("sharp.png", dpi=300)