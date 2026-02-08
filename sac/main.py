import gymnasium as gym
from dataclasses import dataclass
import torch
from torch import nn, optim
from torch.distributions import Normal
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

@dataclass
class Config:
  total_episodes: int = 100
  gamma: float = 0.99
  lr: float = 1e-3
  tau: float = 1e-3
  rb_length: int = 100000
  batch_size: int = 64
  noise_decay: float = 0.9995
  alpha: float = 0.1

  a_lb: np.array = None
  a_ub: np.array = None
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
  def __init__(self, state_dim, action_dim, hidden_dims):
    super(PolicyNetwork, self).__init__()
    self.input_layer = nn.Linear(state_dim, hidden_dims[0])
    self.hidden_layers = nn.ModuleList()
    for i in range(len(hidden_dims)-1):
      self.hidden_layers.append(
        nn.Linear(hidden_dims[i], hidden_dims[i+1])
      )
    self.mu_output_layer = nn.Linear(hidden_dims[-1], action_dim)
    self.std_output_layer = nn.Linear(hidden_dims[-1], action_dim)
    self.relu = nn.ReLU()
    self.softplus = nn.Softplus()
  
  def forward(self, x):
    x = self.relu(self.input_layer(x))
    for i in self.hidden_layers:
      x = self.relu(i(x))
    mu = self.mu_output_layer(x)
    std = self.softplus(self.std_output_layer(x))

    return mu, std


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


class SAC:
  def __init__(self, config):
    self.config = config
    self.online_value_network_1 = ValueNetwork(config.state_dim, config.action_dim, [256, 256])
    self.online_value_network_2 = ValueNetwork(config.state_dim, config.action_dim, [256, 256])
    self.target_value_network_1 = ValueNetwork(config.state_dim, config.action_dim, [256, 256])
    self.target_value_network_2 = ValueNetwork(config.state_dim, config.action_dim, [256, 256])
    
    self.online_policy_network = PolicyNetwork(config.state_dim, config.action_dim, [256, 256])

    self.target_value_network_1.load_state_dict(self.online_value_network_1.state_dict())
    self.target_value_network_2.load_state_dict(self.online_value_network_2.state_dict())

    value_params = list(self.online_value_network_1.parameters()) + list(self.online_value_network_2.parameters())
    self.value_optimizer = optim.Adam(value_params, lr=self.config.lr)
    self.policy_optimizer = optim.Adam(self.online_policy_network.parameters(), lr=self.config.lr)

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
  
  def sample_action(self, s, explore=True):
      mu, std = self.online_policy_network(s)
      if explore:
        normal = Normal(mu, std)
        a_normal = normal.rsample()
        a_squashed = F.tanh(a_normal)

        entropy = normal.log_prob(a_normal) - torch.log(1 - a_squashed.pow(2) + 1e-6)
        entropy = -entropy.sum(dim=1, keepdim=True)

        return a_squashed, entropy
      else:
        a_normal = mu
        a_squashed = F.tanh(a_normal)

        return a_squashed, None


  def optimize(self, batch):
    s = batch.states
    sp = batch.next_states
    r = batch.rewards
    t = batch.terminated
    a = batch.actions

    with torch.no_grad():
      a_squashed, entropy = self.sample_action(sp)

      q_sp_1 = self.target_value_network_1(sp, a_squashed)
      q_sp_2 = self.target_value_network_2(sp, a_squashed)
      min_q_sp = torch.min(q_sp_1, q_sp_2)

      target = r + self.config.gamma * (min_q_sp + self.config.alpha * entropy) * (1 - t)

    q_s_1 = self.online_value_network_1(s, a)
    td_err_1 = (target.detach() - q_s_1)
    value_loss_1 = (td_err_1).pow(2).mul(0.5).mean()
    q_s_2 = self.online_value_network_2(s, a)
    td_err_2 = (target.detach() - q_s_2)
    value_loss_2 = (td_err_2).pow(2).mul(0.5).mean()

    value_loss = value_loss_1 + value_loss_2

    self.value_optimizer.zero_grad()
    value_loss.backward()
    self.value_optimizer.step()

    a_squashed, entropy = self.sample_action(s)

    q_s_1 = self.online_value_network_1(s, a_squashed)
    q_s_2 = self.online_value_network_2(s, a_squashed)
    min_q_s = torch.min(q_s_1, q_s_2)
    policy_loss = -(min_q_s + self.config.alpha * entropy).mean()

    self.policy_optimizer.zero_grad()
    policy_loss.backward()
    self.policy_optimizer.step()

    self.soft_update(self.online_value_network_1, self.target_value_network_1)
    self.soft_update(self.online_value_network_2, self.target_value_network_2)

  @torch.no_grad
  def select_action(self, s, explore=True):
    s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0) 
        
    a_squashed, _ = self.sample_action(s_tensor, explore)

    a_squashed = a_squashed.cpu().numpy()[0]

    return a_squashed


def train(config, sac):
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
      a_squashed = sac.select_action(s)

      a_scaled = config.a_lb + (a_squashed + 1) / 2 * (config.a_ub - config.a_lb)

      s_p, r, terminated, truncated, _ = env.step(a_scaled)

      rewards += r
      rb.insert(s, a_squashed, r, s_p, terminated)

      if not terminated:
        s = s_p

      if rb.cnt >= config.batch_size:
        batch = rb.sample()
        sac.optimize(batch)

      if truncated:
        terminated = True

    print(f"Episode {episode}: {rewards} rewards")

    returns.append((rewards + sum(returns[-min(window_len-1, len(returns)):])) / min(window_len, len(returns)+1))
    line.set_xdata(range(len(returns)))
    line.set_ydata(returns)
    ax.relim()           
    ax.autoscale_view()  
    plt.pause(0.01)

def simulate(config, sac):
  env = gym.make("Pendulum-v1", render_mode='human')

  s, _ = env.reset()
  terminated = False
  rewards = 0
  truncated = False
  
  while not (terminated or truncated):
    a_squashed = sac.select_action(s, False)
    a_scaled = config.a_lb + (a_squashed + 1) / 2 * (config.a_ub - config.a_lb)

    s_p, r, terminated, truncated, _ = env.step(a_scaled)

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
      a_lb=np.array([-2.0]),
      a_ub=np.array([2.0])
  )
  sac = SAC(config)
  train(config, sac)
  simulate(config, sac)