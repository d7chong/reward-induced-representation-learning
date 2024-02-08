import numpy as np
import cv2
from tqdm import tqdm
import time
import wandb
import argparse
import gym
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions import MultivariateNormal

from sprites_datagen.rewards import *
from sprites_env import *

from baselines import ActorCritic

from general_utils import *

"""
[Psuedocode]
    for epoch in range(num_epochs):
        for t in range(local_steps_per_epoch):
            policy_rollout()
            calculate_advantages()
        
        for t in range(update_epochs):
            update_policy() # calculating loss
                policy_optimizer.zero_grad()
                value_optimizer.zero_grad()
                
                total_loss = l_clip() + l_vf() + l_entropy()
                
                total_loss.backward()
                
                policy_optimizer.step()
                value_optimizer.step()     
"""

# ignore warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class PPO:
    def __init__(self, baseline_class, env, env_type, use_gpu, wandb, **hyperparameters):
        self._init_hyperparameters(hyperparameters)
        set_seed()
        
        self.device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
        self.wandb = wandb
        
        self.mode = baseline_class
        self.env_type = env_type
        
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
        self.num_distractors = env.spec.id[-1]
        
        # Select Baseline
        self.actor_critic = ActorCritic(env.observation_space, env.action_space, mode=self.mode).to(self.device)
        
        # Load weights
        # self.actor_critic.load_state_dict(torch.load(f"weights/{self.mode}-nd-{self.num_distractors}-eps-400.pth"))
        
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.lr)
        
        self.max_grad_norm = 0.5
    
    def learn(self, total_timesteps):
        print(f"TRAINING WITH BASELINE CLASS: {self.mode}")
        
        self.t = 0 # global timesteps
        episode = 1
        
        while self.t < total_timesteps:
            start_time = time.time()
            
            # Policy Rollout
            batch_obs, batch_actions, batch_log_probs, batch_rewards, batch_rtgs, batch_lens = self.policy_rollout()
            batch_obs, batch_actions, batch_log_probs = batch_obs.to(self.device), batch_actions.to(self.device), batch_log_probs.to(self.device)
            batch_rewards, batch_rtgs, batch_lens = batch_rewards.to(self.device), batch_rtgs.to(self.device), batch_lens.to(self.device)
            
            # Calculate Advantages
            with torch.no_grad():
                values, _, _ = self.actor_critic.evaluate_actions(batch_obs, batch_actions)
                values = values.view(-1).to(self.device)
            A_k = self.compute_advantages(batch_rtgs, values)
            
            # Policy Update
            for _ in range(self.n_updates_per_iteration):
                self.update_policy(batch_obs, batch_actions, batch_log_probs, batch_rtgs, A_k)
            
            end_time = time.time()
            
            avg_eps_reward = torch.sum(batch_rewards) / torch.sum(batch_lens) * self.max_timesteps_per_episode
            progress_percent = self.t / total_timesteps * 100
            remaining_time = (end_time - start_time) * (total_timesteps - self.t) / self.timesteps_per_batch
            
            # Print progress every episode
            print("TIMESTEPS: {:<6}/{:<6} ({:<6.2f}%) | AVG_EPS_REWARD = {:<6.2f} | REMAINING TIME: {:<6.2f}s".format(
                int(self.t), total_timesteps, progress_percent, avg_eps_reward, remaining_time
            ))
            
            if self.wandb:
                wandb_title = f"Follow ({self.num_distractors} distractor)" if self.num_distractors==1 else f"Follow ({self.num_distractors} distractors)"
                for batch_reward in batch_rewards:
                    # print(batch_reward.size())
                    for step_reward in batch_reward:
                        wandb.log({wandb_title: step_reward * self.max_timesteps_per_episode})
            
            episode += 1
            
            # save weights
            if episode % 100 == 0:
                torch.save(self.actor_critic.state_dict(), f"weights/{self.mode}-nd-{self.num_distractors}-eps-{episode}.pth")

    def policy_rollout(self):
        batch_obs, batch_actions, batch_log_probs, batch_rewards, batch_rtgs, batch_lens = [], [], [], [], [], []
        
        t = 0
        
        while t < self.timesteps_per_batch:
            eps_rewards = []
            obs = self.env.reset()
            done = False
            
            for ep_t in range(self.max_timesteps_per_episode):
                # Render the environment using cv2
                if self.render and t % self.render_every_i == 0:
                    img = self.env.render(mode='rgb_array')
                    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.imshow("environment", img)
                    # save image
                    cv2.imwrite(f"gif_imgs/{self.mode}-nd-{self.num_distractors}-eps-{self.t}.png", img)
                    cv2.waitKey(1)
                
                self.t += 1
                t += 1
                
                with torch.no_grad():
                    if isinstance(obs, np.ndarray):
                        obs = torch.tensor(obs, dtype=torch.float).to(self.device)
                    batch_obs.append(obs.clone())
                    action, _, log_prob = self.actor_critic(obs)
                action, log_prob = action.cpu().numpy(), log_prob.cpu().numpy()
                
                obs, reward, done, _ = self.env.step(action)
                
                # Collect rewards, action, log probs
                eps_rewards.append(reward)
                batch_actions.append(action)
                batch_log_probs.append(log_prob.reshape(-1))
                
                if done:
                    break
                
            # Log rewards in wand for every timestep
            """
            if self.wandb:
                wandb_title = f"Follow ({self.num_distractors} distractor)" if self.num_distractors==1 else f"Follower ({self.num_distractors} distractors)"
                for eps_reward in eps_rewards:
                    wandb.log({wandb_title: eps_reward*(ep_t+1)})
            """
            
            batch_lens.append(ep_t + 1)
            batch_rewards.append(eps_rewards)
            
            
        with torch.no_grad():
            batch_rtgs = self.compute_rtgs(batch_rewards)
            
        batch_obs = torch.cat([obs.unsqueeze(0) for obs in batch_obs], dim=0).to(self.device)
        batch_actions = torch.tensor(batch_actions, dtype=torch.float).to(self.device)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float).to(self.device)
        
        batch_rewards = torch.tensor(batch_rewards, dtype=torch.float).to(self.device)
        batch_lens = torch.tensor(batch_lens, dtype=torch.float).to(self.device)
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float).to(self.device)
            
        return batch_obs, batch_actions, batch_log_probs, batch_rewards, batch_rtgs, batch_lens
    
    
    def update_policy(self, batch_obs, batch_actions, batch_log_probs, batch_rtgs, A_k):
        values, curr_log_probs, dist_entropy = self.actor_critic.evaluate_actions(batch_obs, batch_actions)
        values = values.view(-1)
        
        ratios = torch.exp(curr_log_probs - batch_log_probs)
        ratios = ratios.view(-1)
        
        surr1 = ratios * A_k
        surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k
        
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(batch_rtgs, values)
        entropy_loss = -torch.mean(dist_entropy)
        
        self.optimizer.zero_grad()
        
        total_loss = value_loss * self.value_loss_coef + policy_loss + self.entropy_coef * entropy_loss
        
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
        
        self.optimizer.step()
    
    def compute_rtgs(self, batch_rews):
        batch_rtgs = []
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0
            
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
        
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float32)
        batch_rtgs = (batch_rtgs - batch_rtgs.mean()) / (batch_rtgs.std() + 1e-10)
        
        return batch_rtgs

    def compute_advantages(self, batch_rtgs, values):
        A_k = batch_rtgs - values
        A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
        return A_k
    
    def evaluate(self, batch_obs, batch_actions):
        action_features, values = self.actor_critic(batch_obs)
        
        dist = self.new_dist(action_features)
        
        log_probs = dist.log_probs(batch_actions)
        dist_entropy = dist.entropy().mean()
        
        return values.squeeze(), log_probs, dist_entropy
    
    def _init_hyperparameters(self, hyperparameters):
        self.timesteps_per_batch = 2048
        self.max_timesteps_per_episode = 40
        self.n_updates_per_iteration = 10
        self.gamma = 0.95
        self.clip = 0.2
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        self.lr = 3e-4
        self.render = False
        self.render_every_i = 1
        
        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + str(val))