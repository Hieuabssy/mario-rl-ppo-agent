import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

class NatureCNN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(NatureCNN, self).__init__()
        c, h, w = input_shape
        self.conv1 = nn.Conv2d(c, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            x = self.conv1(dummy)
            x = self.conv2(x)
            x = self.conv3(x)
            linear_input_size = x.view(1, -1).size(1)
            
        self.fc = nn.Linear(linear_input_size, 512)
        self.actor = nn.Linear(512, num_actions)
        self.critic = nn.Linear(512, 1)

    def forward(self, x):
        x = x / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

class PPO:
    def __init__(self, policy, env, verbose=1, tensorboard_log=None, learning_rate=1e-5, n_steps=512, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, batch_size=64, n_epochs=10):
        self.env = env
        
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3:
            if obs_shape[-1] in [1, 3, 4]:
                self.channels_last = True
                self.c, self.h, self.w = obs_shape[-1], obs_shape[0], obs_shape[1]
            else:
                self.channels_last = False
                self.c, self.h, self.w = obs_shape
        else:
            raise ValueError("Observation space must be 3D for CNN")
            
        self.num_actions = env.action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = NatureCNN((self.c, self.h, self.w), self.num_actions).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        
        self.writer = SummaryWriter(tensorboard_log) if tensorboard_log else None
        self.verbose = verbose
        self.num_timesteps = 0

    def preprocess_obs(self, obs):
        if self.channels_last:
            obs = np.transpose(obs, (0, 3, 1, 2))
            
        # Ensure array is contiguous and proper dtype
        obs = np.array(obs, dtype=np.float32)
        return torch.tensor(obs, dtype=torch.float32).to(self.device)

    def predict(self, state, deterministic=False):
        with torch.no_grad():
            state_tensor = self.preprocess_obs(state)
            logits, _ = self.policy(state_tensor)
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                dist = Categorical(logits=logits)
                action = dist.sample()
            return action.cpu().numpy(), None

    def learn(self, total_timesteps, callback=None):
        if callback:
            callback.model = self
            callback.n_calls = self.num_timesteps
        
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
            
        try:
            num_envs = self.env.num_envs
        except AttributeError:
            num_envs = 1
            obs = np.expand_dims(obs, axis=0)
        
        done = np.zeros(num_envs, dtype=bool)
        
        while self.num_timesteps < total_timesteps:
            obs_buffer = []
            actions_buffer = []
            logprobs_buffer = []
            rewards_buffer = []
            dones_buffer = []
            values_buffer = []
            
            for _ in range(self.n_steps):
                obs_t = self.preprocess_obs(obs)
                with torch.no_grad():
                    logits, value = self.policy(obs_t)
                    dist = Categorical(logits=logits)
                    action = dist.sample()
                    logprob = dist.log_prob(action)
                
                next_obs, reward, done, info = self.env.step(action.cpu().numpy())
                if isinstance(next_obs, tuple):
                    next_obs = next_obs[0]
                
                obs_buffer.append(obs)
                actions_buffer.append(action.cpu().numpy())
                logprobs_buffer.append(logprob.cpu().numpy())
                rewards_buffer.append(reward)
                dones_buffer.append(done)
                values_buffer.append(value.squeeze(-1).cpu().numpy())
                
                obs = next_obs
                self.num_timesteps += num_envs
                
                if callback:
                    callback.n_calls = self.num_timesteps
                    callback._on_step()

            obs_t = self.preprocess_obs(obs)
            with torch.no_grad():
                _, next_value = self.policy(obs_t)
                next_value = next_value.squeeze(-1).cpu().numpy()
            
            rewards = np.array(rewards_buffer)
            values = np.array(values_buffer)
            if values.ndim == 1:
                values = np.expand_dims(values, axis=-1)
            dones = np.array(dones_buffer)
            
            returns = np.zeros_like(rewards)
            advantages = np.zeros_like(rewards)
            
            last_gae_lam = 0
            for t in reversed(range(self.n_steps)):
                if t == self.n_steps - 1:
                    next_non_terminal = 1.0 - done
                    next_val = next_value
                else:
                    next_non_terminal = 1.0 - dones[t+1]
                    next_val = values[t+1]
                delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
                advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
                
            returns = advantages + values
            
            b_obs = np.array(obs_buffer).reshape((-1,) + self.env.observation_space.shape)
            b_actions = np.array(actions_buffer).reshape(-1)
            b_logprobs = np.array(logprobs_buffer).reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
            
            dataset_size = self.n_steps * num_envs
            indices = np.arange(dataset_size)
            
            for epoch in range(self.n_epochs):
                np.random.shuffle(indices)
                for start in range(0, dataset_size, self.batch_size):
                    end = start + self.batch_size
                    mb_idx = indices[start:end]
                    
                    mb_obs = self.preprocess_obs(b_obs[mb_idx])
                    mb_actions = torch.tensor(b_actions[mb_idx], dtype=torch.long).to(self.device)
                    mb_logprobs = torch.tensor(b_logprobs[mb_idx], dtype=torch.float32).to(self.device)
                    mb_advantages = torch.tensor(b_advantages[mb_idx], dtype=torch.float32).to(self.device)
                    mb_returns = torch.tensor(b_returns[mb_idx], dtype=torch.float32).to(self.device)
                    
                    logits, mb_values = self.policy(mb_obs)
                    mb_values = mb_values.squeeze(-1)
                    dist = Categorical(logits=logits)
                    new_logprobs = dist.log_prob(mb_actions)
                    entropy = dist.entropy().mean()
                    
                    logratio = new_logprobs - mb_logprobs
                    ratio = torch.exp(logratio)
                    
                    pg_loss1 = mb_advantages * ratio
                    pg_loss2 = mb_advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                    pg_loss = -torch.min(pg_loss1, pg_loss2).mean()
                    
                    v_loss = F.mse_loss(mb_values, mb_returns)
                    
                    loss = pg_loss + self.vf_coef * v_loss - self.ent_coef * entropy
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.optimizer.step()
            
            if self.writer:
                self.writer.add_scalar("rollout/ep_rew_mean", rewards.sum(axis=0).mean(), self.num_timesteps)
                self.writer.add_scalar("train/loss", loss.item(), self.num_timesteps)

    def save(self, path):
        if not path.endswith('.pth'):
            path += '.pth'
        torch.save(self.policy.state_dict(), path)
        
    @classmethod
    def load(cls, path, env=None, **kwargs):
        if not path.endswith('.pth'):
            path += '.pth'
            
        if env is None:
            raise ValueError("Env must be provided to load model")
            
        instance = cls('CnnPolicy', env, **kwargs)
        instance.policy.load_state_dict(torch.load(path, map_location=instance.device))
        return instance
