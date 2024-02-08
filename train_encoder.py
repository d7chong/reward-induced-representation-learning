import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sprites_datagen.moving_sprites import MovingSpriteDataset
from sprites_datagen.rewards import *
from model import *
from general_utils import *

def train_reward_predictor(spec):
    print("\nMODEL:  \tREWARD PREDICTOR")
    print(f"DEVICE: \t{(spec.device).upper()}\n")
    
    reward_tag = spec.rewards
    spec.rewards = reward_list(spec.rewards)
    
    model = RewardPredictor(rewards=spec.rewards)
    model.init_weights()
    
    # Generate random trajectories
    train_data = MovingSpriteDataset(spec)
    train_dataloader = DataLoader(train_data, spec.batch_size, shuffle=False)
    
    # Set up optimizer and loss function
    optimizer = optim.RAdam(model.parameters(), lr=1e-3)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    criterion = nn.MSELoss()
    
    progress_bar = tqdm(total=spec.num_epochs, desc='TRAINING REWARD PREDICTOR', position=0)

    lowest_loss = float("inf")
    
    model.train()
    for epoch in range(1, spec.num_epochs+1):
        epoch_loss = 0

        for batch in train_dataloader:
            optimizer.zero_grad()
        
            images = batch["images"].squeeze(0)[:spec.N, :, :, :]
            images = torch.cat([images, torch.zeros((spec.T - images.size(0), *images.size()[1:]))]) # zero padding
            
            rewards = torch.stack([batch["rewards"][r] for r in batch["rewards"].keys()]).squeeze(1)
            
            predicted_rewards = model(images).to('cpu')
            
            loss = criterion(predicted_rewards, rewards)
            
            epoch_loss += loss.item()
            
            loss.backward() # compute gradients
            optimizer.step() # update weights
        lr_scheduler.step() # update learning rate
            
        epoch_loss /= len(train_dataloader)
        
        progress_bar.set_postfix(loss=epoch_loss)
        progress_bar.update(1)
        
        if epoch_loss < lowest_loss:
            lowest_loss = epoch_loss
            
            reward_predictor_tag = f"ns-{spec.shapes_per_traj}-e-{spec.num_epochs}-nt-{spec.num_trajectories}-rew-{reward_tag}.pt"
            torch.save(model.encoder.state_dict(), f"weights/reward-predictor-encoder-{reward_predictor_tag}")
            torch.save(model.mlp.state_dict(), f"weights/reward-predictor-mlp-{reward_predictor_tag}")
            torch.save(model.lstm.state_dict(), f"weights/reward-predictor-lstm-{reward_predictor_tag}")
        
    progress_bar.close()

def train_image_reconstruction(spec):
    print("\nMODEL:  \tIMAGE RECONSTRUCTION")
    print(f"DEVICE: \t{(spec.device).upper()}\n")
    
    reward_tag = spec.rewards
    spec.rewards = reward_list(spec.rewards)
    
    encoder = Encoder()
    decoder = Decoder()
    
    encoder.init_weights()
    decoder.init_weights()
    
    # Generate random trajectories
    train_data = MovingSpriteDataset(spec)
    train_dataloader = DataLoader(train_data, spec.batch_size, shuffle=False)
    
    # Set up optimizer and loss function
    optimizer = optim.RAdam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    criterion = nn.MSELoss()
    
    progress_bar = tqdm(total=spec.num_epochs, desc='TRAINING IMAGE RECONSTRUCTION', position=0)

    lowest_loss = float("inf")
    
    encoder.train()
    decoder.train()
    for epoch in range(1, spec.num_epochs+1):
        epoch_loss = 0

        for batch in train_dataloader:
            optimizer.zero_grad()
        
            images = batch["images"].squeeze(0) # train on all images
            
            output = encoder(images)
            output = decoder(output)
            
            loss = criterion(images, output.to('cpu'))
            
            epoch_loss += loss.item()
            
            loss.backward() # compute gradients
            optimizer.step() # update weights
        lr_scheduler.step() # update learning rate
            
        epoch_loss /= len(train_dataloader)
        
        progress_bar.set_postfix(loss=epoch_loss)
        progress_bar.update(1)
        
        if epoch_loss < lowest_loss:
            lowest_loss = epoch_loss
            
            image_reconstruction_tag = f"ns-{spec.shapes_per_traj}-e-{spec.num_epochs}-nt-{spec.num_trajectories}.pt"
            torch.save(encoder.state_dict(), f"weights/image-reconstruction-encoder-{image_reconstruction_tag}")
            
    progress_bar.close()
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training the encoder')
    
    # train or eval
    parser.add_argument('--train', type=int, default=1, help='train or just create figure 2')
    
    # commmon args
    parser.add_argument('--rewards', type=str, default='all', help='rewards to predict')
    parser.add_argument('--wandb', type=int, default=0, help='Use wandb')
    parser.add_argument('--gpu', type=int, default=0, help='Use gpu')
    
    # different args
    parser.add_argument('--encoder_num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--encoder_num_trajectories', type=int, default=100, help='number of trajectories')
    parser.add_argument('--decoder_num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--decoder_num_trajectories', type=int, default=100, help='number of trajectories')
    
    # probs won't change
    parser.add_argument('--T', type=int, default=26, help='maximum sequence length')
    parser.add_argument('--N', type=int, default=5, help='initial time window size')
    parser.add_argument('--shapes_per_traj', type=int, default=2, help='number of shapes per trajectory')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    
    args = parser.parse_args()
    
    set_seed()
    
    encoder_spec = AttrDict(
        resolution=64,  # R
        max_speed=0.05,
        obj_size=0.2,
        num_epochs=args.encoder_num_epochs,
        num_trajectories=args.encoder_num_trajectories,
        shapes_per_traj=args.shapes_per_traj,
        rewards=args.rewards,
        batch_size=args.batch_size,
        N=args.N,
        T=args.T,
        wandb=args.wandb,
        device="cuda" if (torch.cuda.is_available() and args.gpu) else "cpu"
    )
    
    # train_reward_predictor(encoder_spec)
    
    train_image_reconstruction(encoder_spec)