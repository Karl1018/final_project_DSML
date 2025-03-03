# Torch version of power spectrum distance calculation

import torch
import numpy as np
from scipy.ndimage import gaussian_filter1d
from matplotlib import pyplot as plt

SMOOTHING_SIGMA = 5

# Compute power spectra distances and average across all dimensions
def power_spectrum_error(x_gen, x_true):
    pse_errors_per_dim, spectrum_true, spectrum_gen = power_spectrum_error_per_dim(x_gen, x_true)
    return torch.tensor(pse_errors_per_dim).mean(dim=0), spectrum_true, spectrum_gen

def compute_power_spectrum(x):
    fft_real = torch.fft.rfft(x)
    ps = torch.abs(fft_real)**2
    ps_smoothed = torch.tensor(gaussian_filter1d(ps.cpu().numpy(), SMOOTHING_SIGMA), device=x.device)
    return ps_smoothed

def get_average_spectrum(x):
    x_ = (x - x.mean()) / x.std()  # normalize individual trajectories
    spectrum = compute_power_spectrum(x_)
    return spectrum / spectrum.sum()

def power_spectrum_error_per_dim(x_gen, x_true):
    assert x_true.shape[1] == x_gen.shape[1]
    assert x_true.shape[2] == x_gen.shape[2]
    dim_x = x_gen.shape[2]
    pse_per_dim = []
    for dim in range(dim_x):
        spectrum_true = get_average_spectrum(x_true[:, :, dim])
        spectrum_gen = get_average_spectrum(x_gen[:, :, dim])
        hd = hellinger_distance(spectrum_true, spectrum_gen)
        pse_per_dim.append(hd)
    return pse_per_dim, spectrum_true, spectrum_gen

def hellinger_distance(p, q):
    return 1 / torch.sqrt(torch.tensor(2.0)) * torch.sqrt(torch.sum((torch.sqrt(p) - torch.sqrt(q))**2))

# Functions for smoothing power spectra with a Gaussian kernel
def kernel_smoothen(data, kernel_sigma=1):
    """
    Smoothen data with Gaussian kernel
    @param kernel_sigma: standard deviation of gaussian, kernel_size is adapted to that
    @return: internal data is modified but nothing returned
    """
    kernel = get_kernel(kernel_sigma)
    data_final = data.clone()
    data_conv = torch.conv1d(data.view(1, 1, -1), torch.tensor(kernel).view(1, 1, -1), padding=len(kernel)//2)
    pad = int(len(kernel) / 2)
    data_final[:] = data_conv[0, 0, pad:-pad]
    data = data_final
    return data

def gauss(x, sigma=1):
    return 1 / torch.sqrt(torch.tensor(2 * np.pi * sigma ** 2)) * torch.exp(-1 / 2 * (x / sigma) ** 2)

def get_kernel(sigma):
    size = sigma * 10 + 1
    kernel = list(range(size))
    kernel = [float(k) - int(size / 2) for k in kernel]
    kernel = [gauss(torch.tensor(k), sigma).item() for k in kernel]
    kernel = [k / np.sum(kernel) for k in kernel]
    return kernel

# Example usage
if __name__ == "__main__":
    x_gen = torch.randn(100, 10, 3)  # Example generated data
    x_true = torch.randn(100, 10, 3)  # Example true data

    pse = power_spectrum_error(x_gen, x_true)
    print("Power Spectrum Error:", pse.item())