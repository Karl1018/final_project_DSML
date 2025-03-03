import numpy as np
import torch

from psd_torch import power_spectrum_error

predictions = []
# Load predictions
for i in range(1, 21):
    pred = np.load(f"bonus/round_{i}/trajectory.npy")
    predictions.append(torch.tensor(pred).to("cuda"))

# Compute pairwise power spectrum distances
pse = []
for i in range(20):
    for j in range(i+1, 20):
        pse.append(power_spectrum_error(predictions[i], predictions[j])[0].item())

print(pse)
print(np.mean(pse))