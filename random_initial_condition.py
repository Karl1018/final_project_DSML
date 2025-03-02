import numpy as np
import torch
import matplotlib.pyplot as plt

from model.transformer_timeseries import TimeSeriesTransformer
from model.inference import T_step_forecast
from psd_torch import power_spectrum_error

# Load model
model = TimeSeriesTransformer(
    input_size=3,
    enc_len=15,
    batch_first=True,
    num_predicted_features=3
    )

# Load model weights
model.load_state_dict(torch.load("ex4/best_model.pth"))

# Inference
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Initial condition
test_data_path = "data/lorenz63_test.npy"
test_data = torch.tensor(np.load(test_data_path)).to(device)

x = torch.rand(1, 1, 3).to(device)
dec_input = torch.rand(1, 1, 3).to(device)

# Generate the trajectory
trajectory = T_step_forecast(
    model=model,
    enc_len=15,
    dec_len=3,
    initial_enc_input=x,
    initial_dec_input=dec_input,
    T=len(test_data),
    device=device
    )

# PSD
psd_error, spectrum_true, spectrum_gen  = power_spectrum_error(trajectory, test_data.unsqueeze(0))
print("PSD error: ", psd_error)

# Plot and save the PSD
ax1 = plt.subplot(111)
ax1.plot(spectrum_true.squeeze().cpu().detach().numpy(), label="True")
ax1.plot(spectrum_gen.squeeze().cpu().detach().numpy(), label="Generated")
ax1.set_title("Power Spectrum Density")
ax1.set_yscale("log")
ax1.set_xscale("log")
ax1.legend()

# Save the PSD plot
plt.savefig("random_initial_condition/psd.png")


# Plot the 3D trajectory
fig = plt.figure()
ax2 = fig.add_subplot(111, projection='3d')
ax2.plot(trajectory[0, :, 0].cpu().detach().numpy(), trajectory[0, :, 1].cpu().detach().numpy(), trajectory[0, :, 2].cpu().detach().numpy(),
        linewidth=0.1
         )
ax2.set_title("Generated 3D Trajectory")

# Save the 3D plot
plt.savefig("random_initial_condition/trajectory.png", dpi=300)