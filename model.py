import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sprites_datagen.rewards import *


class Encoder(nn.Module):
    def __init__(self, input_dim=64, output_dim=64):
        super(Encoder, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=2, padding=1),       
            nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=2, padding=1),       
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1),      
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),     
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),       
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        self.fc = nn.Linear(in_features=128, out_features=output_dim)
    
    def init_weights(self):
        # Initialize weights for the encoder
        for layer in self.modules():
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        x = self.conv(x)                
        x = x.view(x.size(0), -1)
        x = self.fc(x)                  
        return x

class Decoder(nn.Module):
    def __init__(self, input_dim=64, output_dim=64):
        super(Decoder, self).__init__()
        
        self.fc = nn.Linear(input_dim, 128)
        
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),     
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=4, out_channels=1, kernel_size=4, stride=2, padding=1),
            nn.ReLU()        
        )
    
    def init_weights(self):
        # Initialize weights for the decoder
        for layer in self.modules():
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = x.view(x.size(0), -1, 1, 1)
        x = self.deconv(x)
        return x

class MLP(nn.Module):
    def __init__(self, input_dim=64, output_dim=64):
        super(MLP, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def init_weights(self):
        # Initialize weights for the MLP
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        return x

class LSTM(nn.Module): # check model architecture again
    def __init__(self, input_dim=64, hidden_dim=64, num_layers=1):
        super(LSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        
    def forward(self, x):
        x, _ = self.lstm(x)
        return x

class RewardPredictor(nn.Module):
    def __init__(self, 
                 input_dim=64, 
                 hidden_dim=64,
                 rewards=[HorPosReward]
                 ):
        super(RewardPredictor, self).__init__()
        
        self.encoder = Encoder(input_dim=input_dim, output_dim=hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim)
        self.lstm = LSTM(hidden_dim, hidden_dim)
        self.reward_heads = [MLP(hidden_dim, 1) for _ in range(len(rewards))]

    def init_weights(self):
        # Initialize weights for the encoder
        for layer in self.encoder.modules():
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        # Initialize weights for the MLP
        for layer in self.mlp.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        # Initialize weights for the reward heads (MLPs)
        for reward_head in self.reward_heads:
            for layer in reward_head.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
    
    def to_device(self, device):
        self.encoder.to(device)
        self.mlp.to(device)
        self.lstm.to(device)
        for reward_head in self.reward_heads:
            reward_head.to(device)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.mlp(x)
        x = self.lstm(x)
        # print(x.size()) # (T, 64)
        
        rewards = [] # size should be (T, num_reward_heads)
        
        # Input hidden states of each trajectory into each reward head
        for reward_head in self.reward_heads:
            reward = reward_head(x) # x: (T, 64) -> reward: (T, 1)
            rewards.append(reward)
        
        rewards = torch.stack(rewards) # (T, num_reward_heads)
        rewards = rewards.squeeze(-1) 
        
        return rewards

class ImageReconstructor(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=64, output_dim=64,
                 rewards=[AgentXReward, AgentYReward, TargetXReward, TargetYReward]):
        super(ImageReconstructor, self).__init__()
        
        self.encoder = Encoder(input_dim, hidden_dim)
        self.decoder = Decoder(hidden_dim, output_dim)
        
    def init_weights(self):
        # Initialize weights for the encoder
        for layer in self.encoder.modules():
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        # Initialize weights for the decoder
        for layer in self.decoder.modules():
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__ == "__main__":
    # Testing if the network is working
    seq_len = 30
    
    model = RewardPredictor(64, seq_len)
    encoder = Encoder(64, 64)
    
    input = torch.randn(30, 1, 64, 64)
    
    model_output = model(input) # (num_rewards, T)
    
    print(f"model_output size: {model_output.size()}")