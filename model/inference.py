"""
Code for running inference with transformer
"""

import torch.nn as nn 
import torch

from tqdm import tqdm

import model.utils as utils

def T_step_prediction(
    model: nn.Module, dec_len: int, initial_enc_input: torch.Tensor, initial_dec_input: torch.Tensor, T: int, device
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
        # if enc_input.shape[1] > 20:
        #     # enc_input = enc_input[:, -20:, :]
        #     dim_b = 20

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
            
        enc_input = torch.cat((enc_input[:, 1:, :], pred), dim=1)
        dec_input = torch.cat((dec_input, pred), dim=1)
        traj = torch.cat((traj, pred), dim=1)

    return traj[:, 1:, :]  # Remove initial condition

def generate_forecast(model, encoder_input, prediction_length):
    """
    Generate a future trajectory using autoregressive decoding.

    Args:
        model: Trained transformer model
        encoder_input: Tensor of shape (batch_size, encoder_seq_len, input_dim)
        prediction_length: Number of future steps to predict

    Returns:
        Tensor of shape (batch_size, prediction_length, output_dim)
    """
    model.eval()
    with torch.no_grad():
        batch_size = encoder_input.shape[0]

        # Encode the input sequence
        encoder_output = model.encode(encoder_input)

        # Initialize decoder input (e.g., zeros, last known value, or start token)
        decoder_input = torch.zeros(batch_size, 1, encoder_input.shape[-1]).to(encoder_input.device)

        predictions = []

        for _ in range(prediction_length):
            # Decode the next step
            output = model.decode(encoder_output, decoder_input)

            # Take the last predicted time step
            next_step = output[:, -1:, :]  # Shape: (batch_size, 1, output_dim)
            predictions.append(next_step)

            # Append predicted value to decoder input for the next step
            decoder_input = torch.cat([decoder_input, next_step], dim=1)

        return torch.cat(predictions, dim=1)  # Shape: (batch_size, prediction_length, output_dim)