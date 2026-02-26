import gymnasium as gym
from dataclasses import dataclass
import torch
from torch import nn, optim
from torch.distributions import Normal, Categorical
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class Config:
  env: str = "Pendulum-v1"
  is_continuous: bool = True
  n_workers: int = 16
  state_dim: int = 3
  action_dims: np.array = np.array([1])
  a_lb: np.array = np.array([-2.0])
  a_ub: np.array = np.array([2.0])

  total_episodes: int = 20
  rollout_steps: int = 512
  epochs_per_it: int = 10
  gamma: float = 0.99
  lambd: float = 0.95
  lr: float = 1e-3
  tau: float = 1e-3
  batch_size: int = 64
  ent_coef: float = 0.01
  eps: float = 0.2


@dataclass
class Batch:
  states: torch.Tensor
  actions: torch.Tensor
  rewards: torch.Tensor
  next_states: torch.Tensor
  terminated: torch.Tensor
  values: torch.Tensor
  next_values: torch.Tensor
  log_probs: torch.Tensor
  advantages: torch.Tensor


class ValueNetwork(nn.Module):
  def __init__(self, state_dim, hidden_dims):
    super(ValueNetwork, self).__init__()
    self.input_layer = nn.Linear(state_dim, hidden_dims[0])
    self.hidden_layers = nn.ModuleList()
    for i in range(len(hidden_dims) - 1):
      self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
    self.head = nn.Linear(hidden_dims[-1], 1)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.relu(self.input_layer(x))
    for i in self.hidden_layers:
      x = self.relu(i(x))
    x = self.head(x)

    return x


class PolicyNetwork(nn.Module):
  def __init__(self, state_dim, action_dims, hidden_dims, is_continuous=False):
    super(PolicyNetwork, self).__init__()
    self.is_continuous = is_continuous
    self.input_layer = nn.Linear(state_dim, hidden_dims[0])
    self.hidden_layers = nn.ModuleList()
    for i in range(len(hidden_dims) - 1):
      self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))

    if self.is_continuous:
      self.mu_head = nn.Linear(hidden_dims[-1], sum(action_dims))
      self.std_head = nn.Linear(hidden_dims[-1], sum(action_dims))
    else:
      self.heads = nn.ModuleList(
        [nn.Linear(hidden_dims[-1], dim) for dim in action_dims]
      )

    self.relu = nn.ReLU()
    self.softplus = nn.Softplus()

  def forward(self, x):
    x = self.relu(self.input_layer(x))
    for i in self.hidden_layers:
      x = self.relu(i(x))

    if self.is_continuous:
      mu = self.mu_head(x)
      std = self.softplus(self.std_head(x))

      return mu, std
    else:
      return [head(x) for head in self.heads]


class EpisodeBuffer:
  def __init__(self, batch_size):
    self.batch_size = batch_size
    self.reset()

  def reset(self):
    self.cnt = 0
    self.states = None
    self.actions = None
    self.rewards = None
    self.next_states = None
    self.values = None
    self.next_values = None
    self.log_probs = None
    self.terminated = None
    self.advantages = None

  def concat_by_worker(self, current, new):
    new = np.expand_dims(new, axis=1)

    if current is None:
      current = new
    else:
      current = np.concatenate([current, new], axis=1)

    return current

  def insert(self, s, a, r, s_p, v, v_p, log_prob, terminated):
    self.cnt += s.shape[0]
    self.states = self.concat_by_worker(self.states, s)
    self.actions = self.concat_by_worker(self.actions, a)
    self.rewards = self.concat_by_worker(self.rewards, r)
    self.next_states = self.concat_by_worker(self.next_states, s_p)
    self.values = self.concat_by_worker(self.values, v)
    self.next_values = self.concat_by_worker(self.next_values, v_p)
    self.log_probs = self.concat_by_worker(self.log_probs, log_prob)
    self.terminated = self.concat_by_worker(self.terminated, terminated)

  def set_advantages(self, advantages):
    self.advantages = advantages

  def prepare_train(self):
    self.states = self.states.reshape(-1, self.states.shape[-1])
    self.next_states = self.next_states.reshape(-1, self.next_states.shape[-1])
    self.actions = self.actions.reshape(-1, self.actions.shape[-1])
    self.log_probs = self.log_probs.reshape(-1, 1)
    self.advantages = self.advantages.reshape(-1, 1)
    self.values = self.values.reshape(-1, 1)
    self.next_values = self.next_values.reshape(-1, 1)

    self.rewards = self.rewards.reshape(-1)
    self.terminated = self.terminated.reshape(-1)

  def iter_batches(self):
    indices = np.arange(self.cnt)
    np.random.shuffle(indices)

    for start in range(0, self.cnt, self.batch_size):
      end = min(self.cnt, start + self.batch_size)
      batch_idx = indices[start:end]

      yield Batch(
        states=torch.tensor(self.states[batch_idx], dtype=torch.float32),
        actions=torch.tensor(self.actions[batch_idx], dtype=torch.long),
        rewards=torch.tensor(self.rewards[batch_idx], dtype=torch.float32),
        next_states=torch.tensor(self.next_states[batch_idx], dtype=torch.float32),
        values=torch.tensor(self.values[batch_idx], dtype=torch.float32),
        next_values=torch.tensor(self.next_values[batch_idx], dtype=torch.float32),
        log_probs=torch.tensor(self.log_probs[batch_idx], dtype=torch.float32),
        terminated=torch.tensor(self.terminated[batch_idx], dtype=torch.bool),
        advantages=torch.tensor(self.advantages[batch_idx], dtype=torch.float32),
      )


class PPO:
  def __init__(self, config):
    self.config = config
    self.value_network = ValueNetwork(config.state_dim, [64, 64])
    self.policy_network = PolicyNetwork(
      config.state_dim, config.action_dims, [64, 64], config.is_continuous
    )

    self.value_optimizer = optim.Adam(
      self.value_network.parameters(), lr=self.config.lr
    )
    self.policy_optimizer = optim.Adam(
      self.policy_network.parameters(), lr=self.config.lr
    )

  def sample_action(self, s, explore=True):
    if self.config.is_continuous:
      mu, std = self.policy_network(s)
      if explore:
        dist = Normal(mu, std)
        a = dist.rsample()
        a_squashed = torch.tanh(a)

        log_prob = dist.log_prob(a).sum(axis=-1)
        correction = torch.log(1 - a_squashed.pow(2) + 1e-6).sum(axis=-1)
        log_prob -= correction

        entropy = dist.entropy()

        return a, a_squashed, log_prob, entropy
      else:
        a = mu
        a_squashed = torch.tanh(a)

        return a, a_squashed, None, None
    else:
      multi_logits = self.policy_network(s)
      actions = []
      log_probs = []
      entropies = []

      for logits in multi_logits:
        dist = Categorical(logits=logits)
        if explore:
          a = dist.sample()
          actions.append(a)
          log_probs.append(dist.log_prob(a))
          entropies.append(dist.entropy())
        else:
          actions.append(torch.argmax(logits, dim=-1))

      if not explore:
        return torch.stack(actions, dim=-1), None, None

      combined_a = torch.stack(actions, dim=-1)
      combined_lp = torch.stack(log_probs, dim=-1).sum(dim=-1)
      combined_ent = torch.stack(entropies, dim=-1).mean(dim=-1)

      return combined_a, combined_lp, combined_ent

  @torch.no_grad
  def get_action(self, s, explore=True):
    s_tensor = torch.tensor(s, dtype=torch.float32)
    if self.config.is_continuous:
      a, a_squashed, log_prob, _ = self.sample_action(s_tensor, explore)
      a = a.cpu().numpy()
      a_squashed = a_squashed.cpu().numpy()

      if explore:
        log_prob = log_prob.cpu().numpy()

      return a, a_squashed, log_prob

    else:
      a, log_prob, _ = self.sample_action(s_tensor, explore)
      a = a.cpu().numpy()

      if explore:
        log_prob = log_prob.cpu().numpy()

      return a, log_prob

  @torch.no_grad
  def get_value(self, s):
    s_tensor = torch.tensor(s, dtype=torch.float32)
    v = self.value_network(s_tensor)
    v = v.cpu().numpy().squeeze(-1)

    return v

  @torch.no_grad
  def compute_advantages(self, values, next_values, rewards, terminated):
    total_steps = len(rewards)
    advantages = np.zeros_like(rewards)
    last_gae = 0

    for t in reversed(range(total_steps)):
      mask = 1 - terminated[t]
      td_delta = rewards[t] + (self.config.gamma * next_values[t] * mask) - values[t]

      advantages[t] = last_gae = td_delta + (
        self.config.gamma * self.config.lambd * mask * last_gae
      )

    return advantages

  def optimize(self, batch):
    if self.config.is_continuous:
      mu, std = self.policy_network(batch.states)
      dist = Normal(mu, std)

      actions_squashed = torch.tanh(batch.actions)
      new_log_probs = dist.log_prob(batch.actions).sum(axis=-1, keepdim=True)
      correction = torch.log(1 - actions_squashed.pow(2) + 1e-6).sum(
        axis=-1, keepdim=True
      )
      new_log_probs -= correction

      entropy = dist.entropy().mean()
    else:
      multi_logits = self.policy_network(batch.states)
      new_log_probs = []
      entropies = []

      for i, logits in enumerate(multi_logits):
        dist = Categorical(logits=logits)
        new_log_probs.append(dist.log_prob(batch.actions[:, i]))
        entropies.append(dist.entropy())

      new_log_probs = torch.stack(new_log_probs, dim=-1).sum(dim=-1, keepdim=True)
      entropy = torch.stack(entropies, dim=-1).mean()

    current_values = self.value_network(batch.states)

    ratio = torch.exp(new_log_probs - batch.log_probs)

    surr1 = ratio * batch.advantages
    surr2 = (
      torch.clamp(ratio, 1.0 - self.config.eps, 1.0 + self.config.eps)
      * batch.advantages
    )
    policy_loss = -torch.min(surr1, surr2).mean() - self.config.ent_coef * entropy
    self.policy_optimizer.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 0.5)
    self.policy_optimizer.step()

    value_targets = batch.advantages + batch.values
    value_loss = F.mse_loss(current_values, value_targets)

    self.value_optimizer.zero_grad()
    value_loss.backward()
    torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), 0.5)
    self.value_optimizer.step()


def make_env():
  return gym.make(config.env)


def train(config, ppo):
  envs = gym.vector.AsyncVectorEnv([make_env for _ in range(config.n_workers)])
  eb = EpisodeBuffer(config.batch_size)

  plt.ion()
  _, ax = plt.subplots()
  (line,) = ax.plot([], [])
  window_len = 30
  returns = []

  for episode in range(config.total_episodes):
    s, _ = envs.reset()

    for _ in range(config.rollout_steps):
      if config.is_continuous:
        a, a_squashed, lp = ppo.get_action(s)
        a_take = a_squashed
      else:
        a, lp = ppo.get_action(s)
        a_take = a

      if not config.is_continuous and a_take.shape[-1] == 1:
        a_take = a_take.squeeze(-1)

      if config.is_continuous:
        a_take = config.a_lb + (a_take + 1) / 2 * (config.a_ub - config.a_lb)

      s_p, r, terminated, _, _ = envs.step(a_take)

      v = ppo.get_value(s)
      v_p = ppo.get_value(s_p)

      eb.insert(s, a, r, s_p, v, v_p, lp, terminated)
      s = s_p

    advantages = []
    for w in range(config.n_workers):
      adv = ppo.compute_advantages(
        eb.values[w], eb.next_values[w], eb.rewards[w], eb.terminated[w]
      )
      advantages.append(adv)

    eb.set_advantages(np.array(advantages))

    eb.prepare_train()

    for _ in range(config.epochs_per_it):
      for batch in eb.iter_batches():
        ppo.optimize(batch)

    eb.reset()

    print(f"Episode: {episode}")

    ret = simulate(config, ppo, False)

    returns.append(
      (ret + sum(returns[-min(window_len - 1, len(returns)) :]))
      / min(window_len, len(returns) + 1)
    )
    line.set_xdata(range(len(returns)))
    line.set_ydata(returns)
    ax.relim()
    ax.autoscale_view()
    plt.pause(0.01)


@torch.no_grad
def simulate(config, ppo, render=False):
  render_mode = "human" if render else None
  env = gym.make(config.env, render_mode=render_mode)

  s, _ = env.reset()
  terminated = False
  rewards = 0
  truncated = False

  while not (terminated or truncated):
    if config.is_continuous:
      _, a, _ = ppo.get_action(s)
    else:
      a, _ = ppo.get_action(s)

    if config.is_continuous:
      a = config.a_lb + (a + 1) / 2 * (config.a_ub - config.a_lb)

    if not config.is_continuous and a.shape[-1] == 1:
      a = a.squeeze(-1)

    s_p, r, terminated, truncated, _ = env.step(a)

    rewards += r

    if not terminated:
      s = s_p

    if truncated:
      terminated = True

  print(f"Simulation: {rewards} rewards")

  return rewards


if __name__ == "__main__":
  config = Config()
  ppo = PPO(config)
  train(config, ppo)
  simulate(config, ppo, render=True)
