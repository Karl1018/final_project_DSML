"""
Code for running inference with transformer
"""

import torch.nn as nn 
import torch

from tqdm import tqdm

import model.utils as utils

def T_step_forecast(
    model: nn.Module, enc_len: int, dec_len: int, initial_enc_input: torch.Tensor, initial_dec_input: torch.Tensor, T: int, device
) -> torch.Tensor:
    """
    Generate a trajectory using autoregressive decoding.

    Args:
        model: Trained transformer model
        dec_len: Number of steps to predict
        initial_enc_input: Tensor of shape (batch_size, encoder_seq_len, input_dim)
        initial_dec_input: Tensor of shape (batch_size, input_dim)
        T: Number of steps to predict
        device: Device to run the model on

    Returns:
        Tensor of shape (batch_size, T, output_dim)
    """
    
    if len(initial_enc_input.shape) < 3:
        initial_enc_input.unsqueeze(0)
    if len(initial_dec_input.shape) < 3:
        initial_dec_input.unsqueeze(0)

    traj = initial_dec_input

    enc_input = initial_enc_input
    dec_input = initial_dec_input
    # print("enc_input shape is {}".format(enc_input.shape))
    # print("dec_input shape is {}".format(dec_input.shape))

    for t in tqdm(range(T), desc="Generating trajectory", unit="step"):
        dim_a = dec_input.shape[1]
        dim_b = enc_input.shape[1]
        if dec_input.shape[1] > dec_len:
            dec_input = dec_input[:, -dec_len:, :]
            dim_a = dec_len
        if enc_input.shape[1] > enc_len:
            enc_input = enc_input[:, -enc_len:, :]
            dim_b = enc_len

        # print("dim_a: {}, dim_b: {}".format(dim_a, dim_b))
        # print("enc_input shape: {}, dec_input shape: {}".format(enc_input.shape, dec_input.shape))
        dec_mask = utils.generate_square_subsequent_mask(
            dim1=dim_a,
            dim2=dim_a,
            device=device
            )

        enc_mask = utils.generate_square_subsequent_mask(
            dim1=dim_a,
            dim2=dim_b,
            device=device
            )
        with torch.no_grad():
            dec_output = model(enc_input, dec_input, enc_mask, dec_mask)
            # print("dec_output shape is {}".format(dec_output.shape))

        pred = dec_output[:, -1, :]
        if len(pred.shape) < 3:
            pred = pred.unsqueeze(0)
            
        enc_input = torch.cat((enc_input, pred), dim=1)
        dec_input = torch.cat((dec_input, pred), dim=1)
        traj = torch.cat((traj, pred), dim=1)

    return traj[:, 1:, :]  # Remove initial condition