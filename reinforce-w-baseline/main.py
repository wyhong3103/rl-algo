import gymnasium as gym
import torch
from torch import nn
from torch import optim
import numpy as np
import matplotlib.pyplot as plt

class ValueDNN(nn.Module):
  def __init__(self, input_dim, hidden_dims, output_dim):
    super(ValueDNN, self).__init__()
    self.input_layer = nn.Linear(input_dim, hidden_dims[0])
    self.hidden_layers = nn.ModuleList()
    for i in range(len(hidden_dims)-1):
      self.hidden_layers.append(
        nn.Linear(hidden_dims[i], hidden_dims[i+1])
      )
    self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
    self.relu = nn.ReLU()
  
  def forward(self, x):
    x = self.relu(self.input_layer(x))
    for i in self.hidden_layers:
      x = self.relu(i(x))
    x = self.output_layer(x)

    return x

class PolicyDNN(nn.Module):
  def __init__(self, input_dim, hidden_dims, output_dim):
    super(PolicyDNN, self).__init__()
    self.input_layer = nn.Linear(input_dim, hidden_dims[0])
    self.hidden_layers = nn.ModuleList()
    for i in range(len(hidden_dims)-1):
      self.hidden_layers.append(
        nn.Linear(hidden_dims[i], hidden_dims[i+1])
      )
    self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
    self.relu = nn.ReLU()
  
  def forward(self, x):
    x = self.relu(self.input_layer(x))
    for i in self.hidden_layers:
      x = self.relu(i(x))
    x = self.output_layer(x)

    return x
  
  def full_pass(self, obs):
    logits = self.forward(obs)

    dist = torch.distributions.Categorical(logits=logits)

    action_idx = dist.sample()

    logpa = dist.log_prob(action_idx).unsqueeze(-1)

    entropy = dist.entropy().unsqueeze(-1)
    
    return action_idx.item(), logpa, entropy
  
  @torch.no_grad()
  def get_action(self, obs, greedy=False):
    logits = self.forward(obs)

    action_idx = 0
    if not greedy:
      dist = torch.distributions.Categorical(logits=logits)
      action_idx = dist.sample().item()
    else:
      action_idx = torch.argmax(logits).item()
    
    return action_idx

class REINFORCE:
  def __init__(self):
    self.model = PolicyDNN(4, [64, 64], 2)
    self.gamma = 0.99
    self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
    self.total_episodes = 500
  
  def optimize(self, rewards, logpas, entropys):
    returns = []
    g_sum = 0
    for r in reversed(rewards):
        g_sum = r + self.gamma * g_sum
        returns.insert(0, g_sum)

    returns = torch.tensor(returns, dtype=torch.float32)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    loss = -(returns * logpas).mean() - 0.01 * entropys.mean()

    self.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
    self.optimizer.step()
    
  def train(self):
    env = gym.make("CartPole-v1")

    plt.ion()  
    _, ax = plt.subplots()
    line, = ax.plot([], []) 
    window_len = 30
    returns = []

    for episode in range(self.total_episodes):
      obs, _ = env.reset()
      terminated = False

      rewards = []
      logpas = []
      entropys = []

      while not terminated:
        action_idx, logpa, entropy = self.model.full_pass(torch.tensor(obs))

        obs_p, reward, terminated, truncated, _ = env.step(action_idx)

        rewards.append(reward)
        logpas.append(logpa)
        entropys.append(entropy)

        if not terminated:
          obs = obs_p
        
        if truncated:
          terminated = True
      
      logpas = torch.stack(logpas).view(-1)
      entropys = torch.stack(entropys).view(-1)
      self.optimize(rewards, logpas, entropys)
      print(f"Episode {episode}: {sum(rewards)} rewards")

      returns.append((sum(rewards) + sum(returns[-min(window_len-1, len(returns)):])) / min(window_len, len(returns)+1))
      line.set_xdata(range(len(returns)))
      line.set_ydata(returns)
      ax.relim()           
      ax.autoscale_view()  
      plt.pause(0.01)
  
  def simulate(self):
    env = gym.make("CartPole-v1", render_mode='human')
    obs, _ = env.reset()
    rewards = 0
    terminated = False

    while not terminated:
      action_idx = self.model.get_action(torch.tensor(obs), True)

      action_idx = action_idx

      obs_p, reward, terminated, truncated, _ = env.step(action_idx)

      rewards += reward

      obs = obs_p

      if (truncated):
        terminated = True
      
    print("Total rewards: ", rewards)

class REINFORCE_W_BASELINE:
  def __init__(self):
    self.policy_model = PolicyDNN(4, [64, 64], 2)
    self.value_model = ValueDNN(4, [64, 64], 1)
    self.gamma = 0.99
    self.policy_optimizer = optim.Adam(self.policy_model.parameters(), lr=1e-3)
    self.value_optimizer = optim.Adam(self.value_model.parameters(), lr=1e-3)
    self.total_episodes = 500
  
  def optimize(self, rewards, logpas, values, entropys):
    returns = []
    g_sum = 0
    for r in reversed(rewards):
        g_sum = r + self.gamma * g_sum
        returns.insert(0, g_sum)

    returns = torch.tensor(returns, dtype=torch.float32)
    value_errors = returns - values

    policy_loss = -(value_errors.detach() * logpas).mean() - 0.01 * entropys.mean()

    self.policy_optimizer.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), max_norm=1.0)
    self.policy_optimizer.step()

    value_loss = value_errors.pow(2).mean()
    self.value_optimizer.zero_grad()
    value_loss.backward()
    torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), max_norm=1.0)
    self.value_optimizer.step()
    
  def train(self):
    env = gym.make("CartPole-v1")

    plt.ion()  
    _, ax = plt.subplots()
    line, = ax.plot([], []) 
    window_len = 30
    returns = []

    for episode in range(self.total_episodes):
      obs, _ = env.reset()
      terminated = False

      rewards = []
      logpas = []
      values = []
      entropys = []

      while not terminated:
        action_idx, logpa, entropy = self.policy_model.full_pass(torch.tensor(obs))
        value = self.value_model(torch.tensor(obs))

        obs_p, reward, terminated, truncated, _ = env.step(action_idx)

        rewards.append(reward)
        logpas.append(logpa)
        values.append(value)
        entropys.append(entropy)

        if not terminated:
          obs = obs_p
        
        if truncated:
          terminated = True
      
      logpas = torch.stack(logpas).view(-1)
      values = torch.stack(values).view(-1)
      entropys = torch.stack(entropys).view(-1)
      self.optimize(rewards, logpas, values, entropys)
      print(f"Episode {episode}: {sum(rewards)} rewards")

      returns.append((sum(rewards) + sum(returns[-min(window_len-1, len(returns)):])) / min(window_len, len(returns)+1))
      line.set_xdata(range(len(returns)))
      line.set_ydata(returns)
      ax.relim()           
      ax.autoscale_view()  
      plt.pause(0.01)
  
  def simulate(self):
    env = gym.make("CartPole-v1", render_mode='human')
    obs, _ = env.reset()
    rewards = 0
    terminated = False

    while not terminated:
      action_idx = self.policy_model.get_action(torch.tensor(obs), True)

      action_idx = action_idx

      obs_p, reward, terminated, truncated, _ = env.step(action_idx)

      rewards += reward

      obs = obs_p

      if (truncated):
        terminated = True
      
    print("Total rewards: ", rewards)

rl = REINFORCE()
rl.train()
rl.simulate()