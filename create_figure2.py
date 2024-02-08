import matplotlib.pyplot as plt
import argparse

import torch
from torch.utils.data import DataLoader

from sprites_datagen.moving_sprites import MovingSpriteDataset

from train_encoder import *
from train_decoder import *
from general_utils import *
from model import *


def create_figure2(spec):
    print("\nCREATING FIGURE 2...")
    
    reward_tag = spec.rewards
    spec.rewards = reward_list(spec.rewards)
    
    # Initialize model modules
    encoder_h, encoder_v = Encoder().to(spec.device), Encoder().to(spec.device) 
    mlp_h, mlp_v = MLP().to(spec.device), MLP().to(spec.device)
    lstm_h, lstm_v = LSTM().to(spec.device), LSTM().to(spec.device)
    decoder_h, decoder_v = Decoder().to(spec.device), Decoder().to(spec.device)
    
    # pretrained weights tags
    encoder_h_tag = f"ns-{spec.shapes_per_traj}-e-{spec.encoder_num_epochs}-nt-{spec.encoder_num_trajectories}-rew-h.pt"
    decoder_h_tag = f"ns-{spec.shapes_per_traj}-e-{spec.decoder_num_epochs}-nt-{spec.decoder_num_trajectories}-rew-h.pt"
    encoder_v_tag = f"ns-{spec.shapes_per_traj}-e-{spec.encoder_num_epochs}-nt-{spec.encoder_num_trajectories}-rew-v.pt"
    decoder_v_tag = f"ns-{spec.shapes_per_traj}-e-{spec.decoder_num_epochs}-nt-{spec.decoder_num_trajectories}-rew-v.pt"
    
    # Load pretrained weights for horizontal reward
    encoder_h.load_state_dict(torch.load(f"weights/reward-predictor-encoder-{encoder_h_tag}"))
    mlp_h.load_state_dict(torch.load(f"weights/reward-predictor-mlp-{encoder_h_tag}"))
    lstm_h.load_state_dict(torch.load(f"weights/reward-predictor-lstm-{encoder_h_tag}"))
    decoder_h.load_state_dict(torch.load(f"weights/decoder-{decoder_h_tag}"))
    
    # Load pretrained weights for vertical reward
    encoder_v.load_state_dict(torch.load(f"weights/reward-predictor-encoder-{encoder_v_tag}"))
    mlp_v.load_state_dict(torch.load(f"weights/reward-predictor-mlp-{encoder_v_tag}"))
    lstm_v.load_state_dict(torch.load(f"weights/reward-predictor-lstm-{encoder_v_tag}"))
    decoder_v.load_state_dict(torch.load(f"weights/decoder-{decoder_v_tag}"))
    
    # Dataset and DataLoader
    test_data = MovingSpriteDataset(spec)
    test_dataloader = DataLoader(test_data, batch_size=spec.batch_size, shuffle=False) # shuffle=False to preserve temporal order
    
    # Get test input images
    batch = next(iter(test_dataloader))
    test_images = batch["images"].squeeze(0)
    test_input = test_images[:spec.N*3, :, :, :]
    test_input = torch.cat([test_input, torch.zeros((spec.T - test_input.size(0), *test_input.size()[1:]))])
    test_input = test_input.to(spec.device)
    
    # Set models to eval mode
    encoder_h.eval(); mlp_h.eval(); lstm_h.eval(); decoder_h.eval()
    encoder_v.eval(); mlp_v.eval(); lstm_v.eval(); decoder_v.eval()
     
    with torch.no_grad():
        output_h = encoder_h(test_input)
        output_h = mlp_h(output_h)  
        output_h = lstm_h(output_h)
        output_h = decoder_h(output_h)
        
        output_v = encoder_v(test_input)
        output_v = mlp_v(output_v)
        output_v = lstm_v(output_v)
        output_v = decoder_v(output_v)
        
    # Preprocess ground truth images
    images_np = test_images.detach().squeeze(0).numpy()
    if images_np.max() != images_np.min():
        images_np = (images_np - images_np.min()) / (images_np.max() - images_np.min())
    images_np = (images_np * 255).astype(np.uint8)
    
    # Preprocess horizontal reward images
    output_h_np = output_h.to("cpu").detach().squeeze(0).numpy()
    if output_h_np.max() != output_h_np.min():
        output_h_np = (output_h_np - output_h_np.min()) / (output_h_np.max() - output_h_np.min() + 1e-8)
    output_h_np = (output_h_np * 255).astype(np.uint8)
    
    # Preprocess vertical reward images
    output_v_np = output_v.to("cpu").detach().squeeze(0).numpy()
    if output_v_np.max() != output_v_np.min():
        output_v_np = (output_v_np - output_v_np.min()) / (output_v_np.max() - output_v_np.min() + 1e-8)
    output_v_np = (output_v_np * 255).astype(np.uint8)
    
    # Visualize input and output
    num_columns = int(spec.T/(spec.N-1))+2
    fig, axs = plt.subplots(3, num_columns, figsize=(num_columns*1.6, 4.5), gridspec_kw={'top': 0.88, 'left': 0.18})
    
    # Make title that doesn't overlap with subplots
    fig.text(0.06, 0.80, "Ground", ha='left', va='center', fontsize=20)
    fig.text(0.06, 0.73, "Truth", ha='left', va='center', fontsize=20)
    fig.text(0.06, 0.49, "Vertical Reward", ha='left', va='center', fontsize=20)
    fig.text(0.06, 0.23, "Horizontal Reward", ha='left', va='center', fontsize=20)
    fig.text(0.18, 0.95, "Time", ha='left', va='center', fontsize=14)
    fig.text(0.18, 0.91, "———⟶", ha='left', va='center', fontsize=18, fontweight='bold')
    
    # col_idx for image
    col_idx = 0
    
    # Set first column
    axs[0, col_idx].axis('off')
    axs[1, col_idx].axis('off')
    axs[2, col_idx].axis('off')
    axs[0, 0].imshow(images_np[0].squeeze(), cmap='gray', vmin=0, vmax=255, aspect='auto')
    
    # Display input and output images as a grid
    for t in range(1, len(images_np)+1, spec.N-1):
        col_idx += 1
        
        axs[0, col_idx].set_title(f'{t}', fontsize=20)
        
        axs[0, col_idx].axis('off')
        axs[1, col_idx].axis('off')
        axs[2, col_idx].axis('off')
        
        axs[0, col_idx].imshow(images_np[t].squeeze(), cmap='gray', vmin=0, vmax=255, aspect='auto')
        axs[1, col_idx].imshow(output_v_np[t].squeeze(), cmap='gray', vmin=0, vmax=255, aspect='auto')
        axs[2, col_idx].imshow(output_h_np[t].squeeze(), cmap='gray', vmin=0, vmax=255, aspect='auto')
    
    # Save the current figure
    plt.savefig(f"imgs/figure_2_reconstructed.jpg")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training the encoder')
    
    # train or eval
    parser.add_argument('--train', type=int, default=1, help='train or just create figure 2')
    
    # commmon args
    parser.add_argument('--rewards', type=str, default='v', help='rewards to predict')
    parser.add_argument('--wandb', type=int, default=0, help='Use wandb')
    parser.add_argument('--gpu', type=int, default=0, help='Use gpu')
    
    # different args
    parser.add_argument('--encoder_num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--encoder_num_trajectories', type=int, default=500, help='number of trajectories')
    parser.add_argument('--decoder_num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--decoder_num_trajectories', type=int, default=50, help='number of trajectories')
    
    # probs won't change
    parser.add_argument('--T', type=int, default=26, help='maximum sequence length')
    parser.add_argument('--N', type=int, default=5, help='initial time window size')
    parser.add_argument('--shapes_per_traj', type=int, default=1, help='number of shapes per trajectory')
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
    
    decoder_spec = AttrDict(
        resolution=64,  # R
        max_speed=0.05,
        obj_size=0.2,
        encoder_num_epochs=args.encoder_num_epochs,
        encoder_num_trajectories=args.encoder_num_trajectories,
        num_epochs=args.decoder_num_epochs,
        num_trajectories=args.decoder_num_trajectories,
        shapes_per_traj=args.shapes_per_traj,
        rewards=args.rewards,
        batch_size=args.batch_size,
        N=args.N,
        T=args.T,
        wandb=args.wandb,
        device="cuda" if (torch.cuda.is_available() and args.gpu) else "cpu"
    )
    
    fig2_spec = AttrDict(
        resolution=64,  # R
        max_speed=0.05,
        obj_size=0.2,
        encoder_num_epochs=args.encoder_num_epochs,
        encoder_num_trajectories=args.encoder_num_trajectories,
        decoder_num_epochs=args.decoder_num_epochs,
        decoder_num_trajectories=args.decoder_num_trajectories,
        num_trajectories=1,
        shapes_per_traj=1,
        rewards=args.rewards,
        batch_size=args.batch_size,
        N=args.N,
        T=args.T,
        device="cuda" if (torch.cuda.is_available() and args.gpu) else "cpu"
    )
    
    if args.train:
        # Train on horizontal reward
        encoder_spec.rewards = 'h'
        decoder_spec.rewards = 'h'
        train_reward_predictor(encoder_spec)
        train_decoder(decoder_spec)
        
        # Train on vertical reward
        encoder_spec.rewards = 'v'
        decoder_spec.rewards = 'v'
        train_reward_predictor(encoder_spec)
        train_decoder(decoder_spec)
    
    # Create figure 2
    create_figure2(fig2_spec)