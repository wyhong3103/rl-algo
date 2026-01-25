import gymnasium as gym
from dataclasses import dataclass
import torch
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt

@dataclass
class Config:
  total_episodes: int = 200
  gamma: float = 0.99
  lr: float = 1e-3
  tau: float = 1e-3
  rb_length: int = 100000
  batch_size: int = 64
  noise_decay: float = 0.9995

  a_lb: list = None
  a_ub: list = None
  state_dim: int = None
  action_dim: int = None


@dataclass
class Batch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    terminated: torch.Tensor


class ValueNetwork(nn.Module):
  def __init__(self, state_dim, action_dim, hidden_dims):
    super(ValueNetwork, self).__init__()
    self.input_layer = nn.Linear(state_dim+action_dim, hidden_dims[0])
    self.hidden_layers = nn.ModuleList()
    for i in range(len(hidden_dims)-1):
      self.hidden_layers.append(
        nn.Linear(hidden_dims[i], hidden_dims[i+1])
      )
    self.output_layer = nn.Linear(hidden_dims[-1], 1)
    self.relu = nn.ReLU()
  
  def forward(self, x, a):
    x = torch.cat((x, a), dim=1)
    x = self.relu(self.input_layer(x))
    for i in self.hidden_layers:
      x = self.relu(i(x))
    x = self.output_layer(x)

    return x


class PolicyNetwork(nn.Module):
  def __init__(self, state_dim, action_dim, hidden_dims, a_lb, a_ub):
    super(PolicyNetwork, self).__init__()
    self.input_layer = nn.Linear(state_dim, hidden_dims[0])
    self.hidden_layers = nn.ModuleList()
    for i in range(len(hidden_dims)-1):
      self.hidden_layers.append(
        nn.Linear(hidden_dims[i], hidden_dims[i+1])
      )
    self.output_layer = nn.Linear(hidden_dims[-1], action_dim)
    self.relu = nn.ReLU()
    self.tanh = nn.Tanh()
    self.a_lb = torch.tensor(a_lb)
    self.a_ub = torch.tensor(a_ub)
  
  def forward(self, x):
    x = self.relu(self.input_layer(x))
    for i in self.hidden_layers:
      x = self.relu(i(x))
    x = self.tanh(self.output_layer(x))

    x = (x + 1) / 2 * (self.a_ub - self.a_lb) + self.a_lb
    return x


class ReplayBuffer:
  def __init__(self, length, batch_size):
    self.length = length
    self.batch_size = batch_size
    self.cnt = 0
    self.s = [None] * self.length
    self.a = [None] * self.length
    self.r = [None] * self.length
    self.s_p = [None] * self.length
    self.terminated = [None] * self.length
  
  def insert(self, s, a, r, s_p, terminated):
    idx = self.cnt % self.length
    self.s[idx] = s
    self.a[idx] = a
    self.r[idx] = r
    self.s_p[idx] = s_p
    self.terminated[idx] = terminated
    self.cnt += 1
  
  def sample(self):
    idx = np.random.randint(0, min(self.cnt, self.length), self.batch_size)
    batch = Batch(
      states=torch.tensor(np.array([self.s[i] for i in idx]), dtype=torch.float32),
      actions=torch.tensor(np.array([self.a[i] for i in idx]), dtype=torch.float32),
      rewards=torch.tensor(np.array([self.r[i] for i in idx]), dtype=torch.float32).unsqueeze(1),
      next_states=torch.tensor(np.array([self.s_p[i] for i in idx]), dtype=torch.float32),
      terminated=torch.tensor(np.array([self.terminated[i] for i in idx]), dtype=torch.float32).unsqueeze(1),
    )
    return batch


class DDPG:
  def __init__(self, config):
    self.config = config
    self.online_value_network = ValueNetwork(config.state_dim, config.action_dim, [256, 256])
    self.target_value_network = ValueNetwork(config.state_dim, config.action_dim, [256, 256])
    
    self.online_policy_network = PolicyNetwork(config.state_dim, config.action_dim, [256, 256], config.a_lb, config.a_ub)
    self.target_policy_network = PolicyNetwork(config.state_dim, config.action_dim, [256, 256], config.a_lb, config.a_ub)

    self.target_value_network.load_state_dict(self.online_value_network.state_dict())
    self.target_policy_network.load_state_dict(self.online_policy_network.state_dict())

    self.value_optimizer = optim.Adam(self.online_value_network.parameters(), lr=self.config.lr)
    self.policy_optimizer = optim.Adam(self.online_policy_network.parameters(), lr=self.config.lr)

    self.noise_scale = 0.1
  
  @torch.no_grad
  def soft_update(self, online_model, target_model):
      for online_p, target_p in zip(
        online_model.parameters(), 
        target_model.parameters()
      ):
          target_p.data.copy_(
            self.config.tau * online_p.data 
            + (1.0 - self.config.tau) * target_p.data
          )
  
  def optimize(self, batch):
    s = batch.states
    sp = batch.next_states
    r = batch.rewards
    t = batch.terminated
    a = batch.actions

    with torch.no_grad():
      argmax_a_sp = self.target_policy_network(sp)
      max_q_sp = self.target_value_network(sp, argmax_a_sp)
      target = r + self.config.gamma * max_q_sp * (1 - t)

    q_s = self.online_value_network(s, a)
    td_err = (target.detach() - q_s)
    value_loss = td_err.pow(2).mul(0.5).mean()

    self.value_optimizer.zero_grad()
    value_loss.backward()
    self.value_optimizer.step()

    argmax_a_s = self.online_policy_network(s)
    max_q_s = self.online_value_network(s, argmax_a_s)
    policy_loss = -max_q_s.mean()

    self.policy_optimizer.zero_grad()
    policy_loss.backward()
    self.policy_optimizer.step()

    self.soft_update(self.online_value_network, self.target_value_network)
    self.soft_update(self.online_policy_network, self.target_policy_network)

  @torch.no_grad
  def select_action(self, s, explore=True):
    s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0) 
        
    a = self.online_policy_network(s_tensor).cpu().numpy()[0]
    
    if explore:
        noise = np.random.normal(0, scale=self.noise_scale, size=a.shape)
        a += noise
        self.noise_scale = max(0.01, self.noise_scale * self.config.noise_decay)
    
    a = np.clip(a, self.config.a_lb, self.config.a_ub)
    return a


def train(config, ddpg):
  env = gym.make("Pendulum-v1")
  rb = ReplayBuffer(config.rb_length, config.batch_size)

  plt.ion()  
  _, ax = plt.subplots()
  line, = ax.plot([], []) 
  window_len = 30
  returns = []

  for episode in range(config.total_episodes):
    s, _ = env.reset()
    terminated = False
    rewards = 0
    truncated = False
    
    while not (terminated or truncated):
      a = ddpg.select_action(s)

      s_p, r, terminated, truncated, _ = env.step(a)

      rewards += r
      rb.insert(s, a, r, s_p, terminated)

      if not terminated:
        s = s_p

      if rb.cnt >= config.batch_size:
        batch = rb.sample()
        ddpg.optimize(batch)

      if truncated:
        terminated = True

    print(f"Episode {episode}: {rewards} rewards")

    returns.append((rewards + sum(returns[-min(window_len-1, len(returns)):])) / min(window_len, len(returns)+1))
    line.set_xdata(range(len(returns)))
    line.set_ydata(returns)
    ax.relim()           
    ax.autoscale_view()  
    plt.pause(0.01)

def simulate(ddpg):
  env = gym.make("Pendulum-v1", render_mode='human')

  s, _ = env.reset()
  terminated = False
  rewards = 0
  truncated = False
  
  while not (terminated or truncated):
    a = ddpg.select_action(s)

    s_p, r, terminated, truncated, _ = env.step(a)

    rewards += r

    if not terminated:
      s = s_p

    if truncated:
      terminated = True

  print(f"Simulation: {rewards} rewards")
  

if __name__ == "__main__":
  config = Config(
      state_dim=3, 
      action_dim=1,
      a_lb=[-2.0],
      a_ub=[2.0]
  )
  ddpg = DDPG(config)
  train(config, ddpg)
  simulate(ddpg)