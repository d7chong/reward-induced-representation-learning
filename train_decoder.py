import argparse

import wandb
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sprites_datagen.moving_sprites import MovingSpriteDataset
from sprites_datagen.rewards import *
from model import Encoder, Decoder, MLP, LSTM
from general_utils import *

import imageio

def train_decoder(spec):
    print("\nMODEL:  \tDECODER")
    print(f"DEVICE: \t{(spec.device).upper()}\n")
    
    reward_tag = spec.rewards
    spec.rewards = reward_list(spec.rewards)
    
    # Dataset and DataLoader
    train_data = MovingSpriteDataset(spec)
    train_dataloader = DataLoader(train_data, spec.batch_size, shuffle=False) # shuffle=False to preserve temporal order
    
    # Initialize models
    encoder = Encoder()
    mlp = MLP()
    lstm = LSTM()
    decoder = Decoder()
    
    # Set weights -> Load pretrained weights / initialize weights
    reward_predictor_tag = f"ns-{spec.shapes_per_traj}-e-{spec.encoder_num_epochs}-nt-{spec.encoder_num_trajectories}-rew-{reward_tag}.pt"
    encoder.load_state_dict(torch.load(f"weights/reward-predictor-encoder-{reward_predictor_tag}"))
    mlp.load_state_dict(torch.load(f"weights/reward-predictor-mlp-{reward_predictor_tag}"))
    lstm.load_state_dict(torch.load(f"weights/reward-predictor-lstm-{reward_predictor_tag}"))
    decoder.init_weights()
    
    # Set model states (train/eval)
    encoder.eval()
    mlp.eval()
    lstm.eval()
    decoder.train()
            
    # Optimizer and loss function
    optimizer = optim.RAdam(decoder.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    lowest_loss = float("inf")
        
    # Use tqdm to visualize training progress
    progress_bar = tqdm(total=spec.num_epochs, desc='TRAINING DECODER', position=0)
    
    for epoch in range(1, spec.num_epochs+1):
        epoch_loss = 0
        
        for batch in train_dataloader:
            optimizer.zero_grad()
            
            images = batch["images"].squeeze(0)
            
            input = images[:spec.N, :, :, :]
            input = torch.cat([input, torch.zeros((spec.T - input.size(0), *input.size()[1:]))])
            
            output = encoder(input) # (T, 1, 64, 64) -> (T, 64)
            output = mlp(output) # (T, 64) -> (T, 64)
            output = lstm(output) # (T, 64) -> (T, 64)
            output = decoder(output) # (T, 64) -> (T, 1, 64, 64)
            
            loss = criterion(images, output)
            
            epoch_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
        # if epoch % 10 == 0:
            # print_temp_imgs(images, output, spec)
        
        epoch_loss /= len(train_dataloader)
        
        progress_bar.set_postfix(loss=epoch_loss)
        progress_bar.update(1)
        
        stop_epoch = epoch
        
        if epoch_loss < lowest_loss:
            lowest_loss = epoch_loss
            
            decoder_tag = f"ns-{spec.shapes_per_traj}-e-{spec.num_epochs}-nt-{spec.num_trajectories}-rew-{reward_tag}.pt"
            torch.save(decoder.state_dict(), f"weights/decoder-{decoder_tag}")
    
    progress_bar.close()
    
    # MAKE IMAGE
    test_data = MovingSpriteDataset(spec)
    test_dataloader = DataLoader(test_data, batch_size=spec.batch_size, shuffle=False)
    batch = next(iter(test_dataloader))
    batch = next(iter(test_dataloader))

    test_images = batch["images"].squeeze(0)
    test_input = test_images[:spec.N, :, :, :]
    test_input = torch.cat([test_input, torch.zeros((spec.T - test_input.size(0), *test_input.size()[1:]))])
    
    # detach
    images_np = test_images.detach().squeeze(0).numpy()
    if images_np.max() != images_np.min():
        images_np = (images_np - images_np.min()) / (images_np.max() - images_np.min())
    images_np = (images_np * 255).astype(np.uint8)
        
    decoder.eval()
    with torch.no_grad():
        output = encoder(test_input)
        output = mlp(output)
        output = lstm(output)
        output = decoder(output)
                
        output_np = output.detach().squeeze(0).numpy()
        
        # rescale to [0, 1] using min-max normalization and avoid division by 0
        if output_np.max() != output_np.min():
            output_np = (output_np - output_np.min()) / (output_np.max() - output_np.min())
        
        # Change dtype to uint8
        output_np = (output_np * 255).astype(np.uint8)
        
        # Visualize input and output
        num_columns = int(spec.T/(spec.N-1))+2
        fig, axs = plt.subplots(2, num_columns, figsize=(num_columns*1.5, 2))
        
        # Display input and output images as a grid
        for col_idx, i in enumerate(range(spec.N, len(images_np), spec.N-1)):
            axs[0, col_idx].set_title(f't={i-(spec.N-1)}')
            
            axs[0, col_idx].imshow(images_np[i].squeeze(), cmap='gray', vmin=0, vmax=255, aspect='auto')
            axs[0, col_idx].axis('off')

            axs[1, col_idx].imshow(output_np[i].squeeze(), cmap='gray', vmin=0, vmax=255, aspect='auto')
            axs[1, col_idx].axis('off')
        
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        
        # save plt figure
        fig.savefig(f"decoder_imgs/sample-image-output.jpg")
        
        plt.close()
            
        plt.show()
    
def print_temp_imgs(images, output, spec):
    import matplotlib.pyplot as plt
    
    images_np = images.detach().squeeze(0).numpy()
    if images_np.max() != images_np.min():
        images_np = (images_np - images_np.min()) / (images_np.max() - images_np.min())
    images_np = (images_np * 255).astype(np.uint8)
    
    # Preprocess horizontal reward images
    output_np = output.detach().squeeze(0).numpy()
    if output_np.max() != output_np.min():
        output_np = (output_np - output_np.min()) / (output_np.max() - output_np.min())
    output_np = (output_np * 255).astype(np.uint8)
    
    # Visualize input and output
    num_columns = int(spec.T/(spec.N-1))+2
    fig, axs = plt.subplots(2, num_columns, figsize=(num_columns*1.6, 3), gridspec_kw={'top': 0.88, 'left': 0.18})
    
    # col_idx for image
    col_idx = 0
    
    # Set first column
    axs[0, col_idx].axis('off')
    axs[1, col_idx].axis('off')
    axs[0, 0].imshow(images_np[0].squeeze(), cmap='gray', vmin=0, vmax=255, aspect='auto')
    
    # Display input and output images as a grid
    for t in range(1, len(images_np)+1, spec.N-1):
        col_idx += 1
        
        axs[0, col_idx].set_title(f'{t}', fontsize=20)
        
        axs[0, col_idx].axis('off')
        axs[1, col_idx].axis('off')
        
        axs[0, col_idx].imshow(images_np[t].squeeze(), cmap='gray', vmin=0, vmax=255, aspect='auto')
        axs[1, col_idx].imshow(output_np[t].squeeze(), cmap='gray', vmin=0, vmax=255, aspect='auto')

    plt.show()