import torch

import gym

import numpy as np
import warnings
import argparse
import wandb

from ppo import PPO
from baselines import ActorCritic

from train_encoder import *

from sprites_datagen.rewards import *
from sprites_env import *

from general_utils import *

# ignore warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def train_baseline(args):
    if args.wandb:
        wandb.login()
        wandb.init(project="clvr-implementation-project", name=args.baseline_class)

    assert args.num_distractors in [0, 1, 2]
    assert args.baseline_class in ['oracle', 'cnn', 'image_scratch', 
                              'image_reconstruction', 'image_reconstruction_finetune', 
                              'reward_prediction', 'reward_prediction_finetune']
    assert args.total_timesteps > 0
    
    # create env_type and env_id
    env_type = 'SpritesState' if args.baseline_class == 'oracle' else 'Sprites'
    env_id = f"{env_type}-v{args.num_distractors}"
    
    assert ('State' in env_id and args.baseline_class == 'oracle') or ('State' not in env_id and args.baseline_class != 'oracle')
    assert env_id in ['Sprites-v0', 'Sprites-v1', 'Sprites-v2', 'SpritesState-v0', 'SpritesState-v1', 'SpritesState-v2']
    
    # Create environment
    env = gym.make(env_id)
    
    # Create PPO agent
    ppo = PPO(baseline_class=args.baseline_class, env=env, env_type=env_type, 
              total_timesteps=args.total_timesteps, use_gpu=args.gpu, wandb=args.wandb)
    
    # Train PPO agent
    ppo.learn(args.total_timesteps)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training baselines')
    
    parser.add_argument('--baseline_class', '-b', type=str, default='oracle', help='baseline class')
    parser.add_argument('--num_distractors', '-n', type=int, default=0, help='number of distractors')
    parser.add_argument('--rewards', '-r', type=str, default='all', help='rewards to predict')
    parser.add_argument('--total_timesteps', '-t', type=int, default=5000000, help='total timesteps')
    parser.add_argument('--encoder_num_epochs', type=int, default=100, help='number of epochs for encoder')
    parser.add_argument('--encoder_num_trajectories', type=int, default=100, help='number of trajectories for encoder')
    parser.add_argument('--wandb', type=int, default=0, help='use wandb')
    parser.add_argument('--gpu', type=bool, default=1, help='use gpu')
    
    args = parser.parse_args()
    
    set_seed()
    
    train_baseline(args)
    
    """"
    import os
    import imageio
    directory = f'C:/Users/d7cho/GitHub/clvr-implementation-project/gif_imgs/'
    
    # convert all images in gif_imgs to gif
    png_files = [f for f in os.listdir(directory) if f.endswith('.png')]
    
    # Create a list to store the image data
    images = []
    
    # make sure images are sorted: the image name is in the format of 'reward_predicton_finetune-ne-eps-NUMBER.png'
    png_files.sort(key=lambda x: int(x.split('-')[-1].split('.')[0]))
    
    # Read each PNG file and append it to the images list
    for png_file in png_files:
        file_path = os.path.join(directory, png_file)
        images.append(imageio.imread(file_path))
    
    # Define the output GIF file path
    output_path = os.path.join(directory, 'output.gif')
    
    # Save the images as a GIF using imageio, with a duration of 0.1 seconds per frame and a loop of 0 (meaning infinite loop)
    imageio.mimsave(output_path, images, duration=0.005, loop=0)
    
    print(f"GIF saved successfully at: {output_path}")
    
    # delete all images in directory
    for f in png_files:
        os.remove(directory + f)
    """