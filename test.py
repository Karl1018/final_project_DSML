import argparse
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

import model.inference as inference
from model.transformer_timeseries import TimeSeriesTransformer
from psd_torch import power_spectrum_error

enc_len = 20
tar_len = 4

# Load model
model = TimeSeriesTransformer(
    input_size=3,
    enc_len=enc_len,
    batch_first=True,
    num_predicted_features=3
    )

# Load model weights
model.load_state_dict(torch.load("model_best.pt"))

# Inference
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Initial condition
test_data_path = "data/lorenz63_test.npy"
test_data = torch.tensor(np.load(test_data_path))
gen_data = torch.tensor(np.load("96/trajectory.npy")).squeeze(0)

src = test_data[0, :].unsqueeze(0).unsqueeze(0).to(device)
print("Initial condition:", src)

# Define parameters
forecast_window = 500
batch_size = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_first = True

# Generate the trajectory
# with torch.no_grad():
#     trajectory = inference.T_step_prediction(
#         model=model,
#         dec_len=tar_len,
#         initial_enc_input=test_data[:enc_len, :].unsqueeze(0).to(device),
#         initial_dec_input=test_data[enc_len-1, :].unsqueeze(0).unsqueeze(0).to(device),
#         T=forecast_window,
#         device=device
#         )

    # trajectory = inference.run_encoder_decoder_inference(
    #     model=model,
    #     src=src,
    #     forecast_window=forecast_window,
    #     batch_size=batch_size,
    #     device=device,
    #     batch_first=batch_first
    # )

# Convert the trajectory to a NumPy array
# trajectory_np = trajectory.squeeze().cpu().numpy()  # Shape: [forecast_window, 3]

# Print the trajectory
# print("Generated Trajectory:")
# print(trajectory_np)

# Visualize the trajectory
# x = trajectory_np[:, 0]
# y = trajectory_np[:, 1]
# z = trajectory_np[:, 2]

x_hat = gen_data[:, 0].cpu().numpy()
y_hat = gen_data[:, 1].cpu().numpy()
z_hat = gen_data[:, 2].cpu().numpy()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.plot(x, y, z)
ax.plot(x_hat, y_hat, z_hat, linewidth=0.1)

# Mark the initial condition
# ax.plot(x_hat[:15], y_hat[:15], z_hat[:15], color='r', linewidth=1, label='Initial Condition')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Generated 3D Trajectory')
plt.legend()
plt.show()

# x_gen = trajectory.to(device)
# x_true = test_data[enc_len: forecast_window + enc_len, :].unsqueeze(0).to(device)
# print(power_spectrum_error(x_gen, x_true))