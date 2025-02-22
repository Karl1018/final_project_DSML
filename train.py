import argparse
import numpy as np
import torch

import model.inference as inference
import model.utils as utils
from model.transformer_timeseries import TimeSeriesTransformer
from model.dataset import TransformerDataset

def train_epoch(
    model,
    training_dataloader,
    optimizer,
    criterion,
    forecast_window,
    enc_seq_len
    ):
    """
    Train the model for one epoch.
    
    Args:
    - model (torch.nn.Module): model to be trained
    - training_dataloader (torch.utils.data.DataLoader): dataloader for training data
    - optimizer (torch.optim.Optimizer): optimizer to be used
    - criterion (torch.nn.Module): loss function to be used
    - forecast_window (int): number of hours to forecast ahead
    - enc_seq_len (int): number of hours to use as input
    
    Returns:
    - None
    """
    # Iterate over all (x,y) pairs in training dataloader
    for i, (src, tgt, tgt_y) in enumerate(training_dataloader):

        # zero the parameter gradients
        optimizer.zero_grad()

        # Generate masks
        tgt_mask = utils.generate_square_subsequent_mask(
            dim1=forecast_window,
            dim2=forecast_window
            )

        src_mask = utils.generate_square_subsequent_mask(
            dim1=forecast_window,
            dim2=enc_seq_len
            )

        # Make forecasts
        prediction = model(src, tgt, src_mask, tgt_mask)

        # Compute and backprop loss
        loss = criterion(tgt_y, prediction)

        loss.backward()

        # Take optimizer step
        optimizer.step()

def validate_epoch(
    model,
    validation_dataloader,
    forecast_window,
    criterion
    ):
    """
    Validate the model for one epoch.
    
    Args:
    - model (torch.nn.Module): model to be validated
    - validation_dataloader (torch.utils.data.DataLoader): dataloader for validation data
    - forecast_window (int): number of hours to forecast ahead
    
    Returns:
    - None
    """

    loss_list = []
    # Iterate over all (x,y) pairs in validation dataloader
    model.eval()

    with torch.no_grad():
    
        for i, (src, _, tgt_y) in enumerate(validation_dataloader):

            prediction = inference.run_encoder_decoder_inference(
                model=model, 
                src=src, 
                forecast_window=forecast_window,
                batch_size=src.shape[1]
                )
            
            loss = criterion(tgt_y, prediction)
            loss_list.append(loss.item())

    print("Validation loss: {}".format(np.mean(loss_list)))

if __name__ == "__main__":
    
    # Arguments processing
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument("-e", "--epochs", type=int, help="Number of epochs", default=10)
    parser.add_argument("--enc", type=int, default=20)
    parser.add_argument("--dec", type=int, default=10)
    parser.add_argument("--tar", type=int, default=10)
    parser.add_argument("--forecast_window", type=int, default=10)
    args = parser.parse_args()

    # Model, optimizer, loss function
    model = TimeSeriesTransformer(
        input_size=3,
        dec_seq_len=args.dec,
        batch_first=True,
        num_predicted_features=3
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    # Dataloaders
    training_data_path = "data/lorenz63_on0.05_train.npy"
    validation_data_path = "data/lorenz63_test.npy"

    training_data = torch.tensor(np.load(training_data_path))
    validation_data = torch.tensor(np.load(validation_data_path))

    SIZE = 100000
    indices = utils.get_indices_input_target(
        num_obs=SIZE,
        input_len=args.enc+args.dec,
        step_size=1,
        forecast_horizon=args.forecast_window,
        target_len=args.tar
        )


    training_dataset = TransformerDataset(
        data=training_data,
        indices=indices,
        enc_seq_len=args.enc,
        dec_seq_len=args.dec,
        target_seq_len=args.tar
        )
    
    validation_dataset = TransformerDataset(
        data=validation_data,
        indices=indices,
        enc_seq_len=args.enc,
        dec_seq_len=args.dec,
        target_seq_len=args.tar
        )
    
    training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=4, shuffle=True)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=4, shuffle=False)

    # Training loop
    for epoch in range(args.epochs):
        train_epoch(
            model=model,
            training_dataloader=training_dataloader,
            optimizer=optimizer,
            criterion=criterion,
            forecast_window=args.forecast_window,
            enc_seq_len=args.enc
            )

        validate_epoch(
            model=model,
            validation_dataloader=validation_dataloader,
            forecast_window=args.forecast_window,
            criterion=criterion
            )
        
