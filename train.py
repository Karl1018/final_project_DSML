import os
import argparse
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

import model.inference as inference
import model.utils as utils
from model.dataset import TransformerDataset
from model.transformer_timeseries import TimeSeriesTransformer
from psd_torch import power_spectrum_error

LOGGER = logging.getLogger("Logger")

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        """
        Args:
            patience (int): Number of epochs to wait before stopping if no improvement.
            min_delta (float): Minimum change in loss to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.wait = 0
        self.early_stop = False

    def __call__(self, current_loss):
        # Check if the loss has improved
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait = 0  # Reset wait counter
        else:
            self.wait += 1  # Increment wait counter

        # Stop training if patience is exceeded
        if self.wait >= self.patience:
            self.early_stop = True

        return self.early_stop, self.best_loss


def train_step(
    model,
    training_dataloader,
    optimizer,
    criterion,
    src_mask,
    tgt_mask,
    debug=False
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

    loss_list = []

    # Iterate over all (x,y) pairs in training dataloader
    for i, (src, tgt, tgt_y) in tqdm(enumerate(training_dataloader), total=len(training_dataloader), desc="Training"):
        # zero the parameter gradients
        optimizer.zero_grad()
        # Make forecasts
        prediction = model(src, tgt, src_mask, tgt_mask)

        # Compute and backprop loss
        loss = criterion(tgt_y, prediction)

        loss_list.append(loss.item())

        loss.backward()

        # Take optimizer step
        optimizer.step()

        # Debugging
        if debug:
            print(f"src: {src}")
            print(f"tgt: {tgt}")
            print(f"tgt_y: {tgt_y}")
            print(f"tgt_mask: {tgt_mask}")
            print(f"src_mask: {src_mask}")
            print(f"prediction: {prediction}")

            raise ValueError("Debugging")
        
    return np.mean(loss_list)

def validate_step(
    model,
    validation_dataloader,
    criterion,
    src_mask,
    tgt_mask,
    ):
    """
    Validate the model for one epoch.
    
    Args:
    - model (torch.nn.Module): model to be validated
    - validation_dataloader (torch.utils.data.DataLoader): dataloader for validation data
    - forecast_window (int): number of hours to forecast ahead
    
    Returns:
    - Average loss over all validation data
    """

    loss_list = []
    # Iterate over all (x,y) pairs in validation dataloader
    model.eval()

    with torch.no_grad():
    
        for i, (src, tgt, tgt_y) in tqdm(enumerate(validation_dataloader), total=len(validation_dataloader), desc="Validation"):

            prediction = model(src, tgt, src_mask, tgt_mask)
            
            loss = criterion(tgt_y, prediction)
            loss_list.append(loss.item())

    return np.mean(loss_list)

def training_loop(model, src_mask, tgt_mask, epochs, optimizer, criterion,training_dataloader, validation_dataloader,
                  device, save_path, debug=False):
    
    loss_list = [float("inf")]
    early_stopping = EarlyStopping(patience=5, min_delta=1e-5)

    # Training loop
    for epoch in range(epochs):
        train_loss = train_step(
            model=model,
            training_dataloader=training_dataloader,
            optimizer=optimizer,
            criterion=criterion,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            debug=debug
            )
        
        print(f"Epoch {epoch + 1}: Training loss = {train_loss}")

        val_loss = validate_step(
            model=model,
            validation_dataloader=validation_dataloader,
            criterion=criterion,
            src_mask=src_mask,
            tgt_mask=tgt_mask
            )
        
        # Keep track of the best model
        if val_loss < min(loss_list):
            best_model = model.state_dict()

        print(f"Epoch {epoch + 1}: Validation loss = {val_loss}")
        loss_list.append(val_loss)

        # Early stopping, patience = 5
        stop, best_loss = early_stopping(val_loss)
        if stop:
            print(f"Early stopping at epoch {epoch + 1} with best loss = {best_loss}")
            LOGGER.info(f"Early stopping at epoch {epoch + 1} with best loss = {best_loss}")
            break

    torch.save(best_model, save_path + "best_model.pth")

def test(model, enc_len, tar_len, test_data, device, save_path, forecast_window=None, save_plot=True):
    """
    Test the model on the test data.
    
    Args:
    - model (torch.nn.Module): model to be tested
    - enc_len (int): encoder sequence length
    - tar_len (int): target sequence length
    - test_data (torch.Tensor): test data
    - device (str): device to use

    Returns:
    - None
    """
    if forecast_window is None: # Forecast the entire test data
        forecast_window = len(test_data) - enc_len

    with torch.no_grad():
        trajectory = inference.T_step_prediction(
            model=model,
            dec_len=tar_len,
            initial_enc_input=test_data[:enc_len, :].unsqueeze(0).to(device),
            initial_dec_input=test_data[enc_len-1, :].unsqueeze(0).unsqueeze(0).to(device),
            T=forecast_window,
            device=device
            )
    # Save trajectory
    np.save(save_path + "trajectory.npy", trajectory.cpu().numpy())

    # Plot power spectrum density error
    
    psd_error, spectrum_true, spectrum_gen = power_spectrum_error(trajectory, test_data[enc_len: forecast_window + enc_len, :].unsqueeze(0))

    if save_plot:
        save_trajectory(trajectory)

        # Plot power spectrum density
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(spectrum_true.squeeze().cpu().numpy(), label="True")
        ax.plot(spectrum_gen.squeeze().cpu().numpy(), label="Generated")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Power")
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_title("Power Spectrum Density")
        ax.legend()
        plt.savefig(save_path + "psd.png")

    print(f"Power spectrum density error: {psd_error}")
    LOGGER.info(f"Power spectrum density error: {psd_error}") 

def save_trajectory(trajectory):
    """
    Save the trajectory generated by the model and the test data.
    
    Args:
    - trajectory (torch.Tensor): trajectory generated by the model
    - test_data (torch.Tensor): test data
    """
    trajectory_np = trajectory.squeeze().cpu().numpy()
    x = trajectory_np[:, 0]
    y = trajectory_np[:, 1]
    z = trajectory_np[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z)
    # ax.plot(x_hat, y_hat, z_hat, marker='x', color='r')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Generated 3D Trajectory')

    plt.savefig(save_path + "trajectory.png")

if __name__ == "__main__":
    # Arguments processing
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument("--epochs", type=int, help="Number of epochs", default=30)
    parser.add_argument("--enc", type=int, default=15)
    parser.add_argument("--dec", type=int, default=3)
    parser.add_argument("--tar", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--random_seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_plot", action="store_true", default=True)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--save_path", type=str, default="bonus/")
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    logging.basicConfig(filename=args.save_path + "log.txt", level=logging.INFO)

    LOGGER.info({"enc_len": args.enc, "dec_len": args.dec, "tar_len": args.tar, "random_seed": args.random_seed, "rounds": args.rounds})

    torch.set_default_device(args.device)
    if args.random_seed:
        torch.manual_seed(args.random_seed)
        print(f"Random seed manually set to {args.random_seed}")


    assert (args.test and args.debug) == False, "Cannot test and debug at the same time"
    if args.debug:
        print("Debugging mode")
        batch_size = 1

    model = TimeSeriesTransformer(
        input_size=3,
        enc_len=args.enc,
        batch_first=True,
        num_predicted_features=3,
        dropout_encoder=0.2,
        dropout_decoder=0.2
        )
    
    if not args.test:

        print("Training...")
 
        batch_size = args.batch_size

        # Data
        training_data_path = "data/lorenz63_on0.05_train.npy"

        training_data = torch.tensor(np.load(training_data_path))
        SIZE = training_data.shape[0]

        # Split data into training and validation
        len_training_data = int(0.8*SIZE)
        len_validation_data = SIZE - len_training_data

        training_data, validation_data = training_data[:len_training_data].to(args.device), training_data[len_training_data:].to(args.device)

        # Datasets and dataloaders
        indices_train = utils.get_indices_entire_sequence(
            num_obs=len_training_data,
            window_size=args.enc+args.dec,
            step_size=1,
            )
        
        indices_validation = utils.get_indices_entire_sequence(
            num_obs=len_validation_data,
            window_size=args.enc+args.dec,
            step_size=1,
            )
        
        training_dataset = TransformerDataset(
            data=training_data,
            indices=indices_train,
            enc_seq_len=args.enc,
            dec_seq_len=args.dec,
            target_seq_len=args.tar,
            )
        
        validation_dataset = TransformerDataset(
            data=validation_data,
            indices=indices_validation,
            enc_seq_len=args.enc,
            dec_seq_len=args.dec,
            target_seq_len=args.tar
            )
        
        training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=args.device))
        validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, generator=torch.Generator(device=args.device))
        # Model, optimizer, loss function

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = torch.nn.MSELoss()
        
        # Generate masks
        tgt_mask = utils.generate_square_subsequent_mask(
            dim1=args.tar,
            dim2=args.tar
            )

        src_mask = utils.generate_square_subsequent_mask(
            dim1=args.tar,
            dim2=args.enc
            )
        
    for i in range(args.rounds):
        print(f"Round {i + 1}")
        LOGGER.info("\n")
        LOGGER.info(f"Round {i + 1}")
        if args.rounds > 1:
            save_path = args.save_path + f"round_{i + 1}/"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
        if not args.test:
            # Training
            training_loop(
                model,
                src_mask=src_mask,
                tgt_mask=tgt_mask,
                epochs=args.epochs,
                optimizer=optimizer,
                criterion=criterion,
                training_dataloader=training_dataloader,
                validation_dataloader=validation_dataloader,
                device=args.device,
                debug=args.debug,
                save_path=save_path
                )
        else:
            model.load_state_dict(torch.load("best_model.pth"))

        # Testing
        print("Testing...")
        test_data_path = "data/lorenz63_test.npy"
        test_data = torch.tensor(np.load(test_data_path), device=args.device)

        test(
            model=model,
            enc_len=args.enc,
            tar_len=args.tar,
            test_data=test_data,
            device=args.device,
            save_plot=args.save_plot,
            save_path=save_path
            )
    
