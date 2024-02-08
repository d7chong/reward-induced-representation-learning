import numpy as np

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from model import Encoder
from distribution import DiagGaussianDistribution


"""
    Class CNN (#1)
        1. cnn <-------------------------------------------------------------------- new CNN architecture
    Class Representation (#2-#6)
        2. image-scratch baseline <------------------------------------------------- from-scratch encoder
        - image-reconstruction 
            → Uses learned representations from the image reconstruction task
            3. image-reconstruction <----------------------------------------------- frozen encoder
            4. image-reconstructon-finetune <--------------------------------------- finetuned encoder
        - reward-prediction
            → Uses learned representations from the reward prediction task
            5. reward-prediction <-------------------------------------------------- frozen encoder
            6. reward-prediction-finetune <----------------------------------------- finetuned encoder 
    Class Oracle (#7)
        7. oracle <----------------------------------------------------------------- oracle
            - (x,y) of agent, target, and distractors
            - upper bound of method
"""

class CNN(nn.Module):
    def __init__(self, observation_space, action_space, output_size, mode='cnn'):
        super(CNN, self).__init__()

        self.w, self.h = observation_space.shape[0], observation_space.shape[1]
        self.output_size = output_size

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),       
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1),      
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1),     
            nn.ReLU(),
            nn.Flatten()
        )

        observation_sample = torch.as_tensor(observation_space.sample()[None, None], dtype=torch.float32)
        num_features = torch.numel(self.encoder(observation_sample))

        self.policy_net = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

        self.value_net = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        obs = obs.view(-1, 1, self.w, self.h)
        x = self.encoder(obs)

        return self.policy_net(x), self.value_net(x)

class Representation(nn.Module):
    def __init__(self, observation_space, action_space, output_size, mode=''):
        super(Representation, self).__init__()

        self.w, self.h = observation_space.shape[0], observation_space.shape[1]
        self.output_size = output_size
        self.mode = mode

        self.encoder = Encoder()
        
        """
        Set the method of training for the encoder:
        [
            image_scratch, 
            image_reconstruction, image_reconstruction_finetune, 
            reward_prediction, reward_prediction_finetune
        ]
        """
        
        # CHANGE THESE WEIGHTS TO RELEVANT WEIGHTS FOR THE ENCODER
        image_reconstruction_weights = 'weights/image-reconstruction-encoder-ns-2-e-100-nt-100.pt'
        reward_prediction_weights = 'weights/reward-predictor-encoder-ns-2-e-100-nt-100-rew-all.pt'
        
        if 'image_reconstruction' in mode:
            self.encoder.load_state_dict(torch.load(image_reconstruction_weights))
        elif 'reward_prediction' in mode:
            self.encoder.load_state_dict(torch.load(reward_prediction_weights))
        else: # image_scratch
            pass
        
        # Freeze encoder weights if method is finetuning
        if 'finetune' not in mode and mode != 'image_scratch':
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.policy_net = nn.Sequential(
            nn.Linear(output_size, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )

        self.value_net = nn.Sequential(
            nn.Linear(output_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def init_weights(self):
        if 'finetune' not in self.mode:
            for layer in self.encoder.modules():
                if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
        
        for layer in self.policy_net.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        
        for layer in self.value_net.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        
    
    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        obs = obs.view(-1, 1, self.w, self.h)
        x = self.encoder(obs)

        return self.policy_net(x), self.value_net(x)

class Oracle(nn.Module):
    def __init__(self, observation_space, action_space, output_size=64, mode='oracle'):
        super(Oracle, self).__init__()
        self.output_size = output_size
        input_size = observation_space.shape[0]

        self.policy_net = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )

        self.value_net = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def init_weights(self):
        for layer in self.policy_net.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        
        for layer in self.value_net.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
    
    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        return self.policy_net(obs), self.value_net(obs)

class ActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, output_size=64, mode=''):
        super(ActorCritic, self).__init__()
        
        if mode == 'cnn':
            self.features_extractor = CNN(observation_space, action_space, output_size, mode)
        elif mode == 'oracle':
            self.features_extractor = Oracle(observation_space, action_space, output_size, mode)
        else:
            self.features_extractor = Representation(observation_space, action_space, output_size, mode)
        
        self.action_dist = DiagGaussianDistribution(action_space.shape[0])
        self.action_net, self.log_std = self.action_dist.proba_distribution_net(latent_dim=self.features_extractor.output_size, log_std_init=0.0)
        
    def init_weights(self):
        self.features_extractor.init_weights()
        
    def load_weights(self, weights_path):
        self.load_state_dict(torch.load(weights_path))
    
    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        policy_net_output, value_net_output = self.features_extractor(obs)
        mean_actions = self.action_net(policy_net_output)
        dist = self.action_dist.proba_distribution(mean_actions, self.log_std)
        actions = dist.sample()
        actions_log_probs = dist.log_prob(actions)
        return actions, value_net_output, actions_log_probs
    
    def evaluate_actions(self, obs, actions):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        policy_net_output, value_net_output = self.features_extractor(obs)
        mean_actions = self.action_net(policy_net_output)
        dist = self.action_dist.proba_distribution(mean_actions, self.log_std)
        actions = actions.view(-1, 2)
        actions_log_probs = dist.log_prob(actions).view(-1, 1)
        return value_net_output, actions_log_probs, dist.entropy()