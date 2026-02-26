import copy
import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from sortedcontainers import SortedList


class DuelingQNetwork(torch.nn.Module):
  def __init__(self, state_size, action_size):
    super(DuelingQNetwork, self).__init__()
    self.fc1 = torch.nn.Linear(state_size, 512)
    self.fc2 = torch.nn.Linear(512, 512)
    self.fc_v = torch.nn.Linear(512, 1)
    self.fc_a = torch.nn.Linear(512, action_size)

  def forward(self, x):
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    v = self.fc_v(x)
    a = self.fc_a(x)
    q = v + a - a.mean(dim=1, keepdim=True)

    return q


class Policy:
  def __init__(self, model, state_size, action_size):
    self.model = model
    self.action_size = action_size
    self.sample_size = state_size
    self.EPS = 1e-1

  def get_action(self, s, get_max=False):
    a_p = self.model(s)
    action_idx = torch.argmax(a_p, dim=-1)

    if not get_max:
      p = torch.rand(action_idx.shape[0])
      mask = p <= self.EPS
      random_actions = torch.randint(0, self.action_size, (mask.sum().item(),))
      action_idx[mask] = random_actions

    return action_idx


class ReplayBuffer:
  def __init__(self, length, batch_size):
    self.length = length
    self.batch_size = batch_size
    self.sl = SortedList()
    self.cnt = 0
    self.alpha = 0.6
    self.experiences = [tuple() for _ in range(length)]

    priority = 1 / (torch.arange(length) + 1)
    self.scaled_priority = priority.pow(self.alpha)
    self.prefix_sum_sp = torch.cumsum(self.scaled_priority, dim=0)

  def insert(self, experience):
    max_priority = self.sl[-1][0] if self.cnt > 0 else 1.0

    idx = self.cnt if self.cnt < self.length else self.sl[0][1]
    if self.cnt >= self.length:
      self.sl.discard(self.sl[0])
    else:
      self.cnt += 1

    experience += (max_priority,)
    self.experiences[idx] = experience
    self.sl.add((experience[-1], idx))

  def update(self, idx, err):
    experience = self.experiences[idx]
    self.sl.discard((experience[-1], idx))

    experience = experience[:-1] + (err,)
    self.sl.add((experience[-1], idx))
    self.experiences[idx] = experience

  def sample(self, beta):
    probs = self.scaled_priority[: self.cnt] / self.prefix_sum_sp[self.cnt - 1]

    ranks = torch.multinomial(probs, self.batch_size)
    indices = [self.sl[-(i + 1)][1] for i in ranks]
    experiences = [self.experiences[i] for i in indices]
    weights = (1 / (self.cnt * probs[ranks])) ** beta
    weights = weights / torch.max(weights)
    return experiences, indices, weights


def train(
  policy,
  batch_size,
  gamma,
  total_episodes,
  replay_buffer_len,
  window_len,
  target_update_freq,
  eps_start=1.0,
  eps_end=0.01,
  eps_decay=0.995,
  tau=0.99,
  beta=0.4,
):
  env = gym.make("CartPole-v1", render_mode=None)
  obs, _ = env.reset()

  target_model = copy.deepcopy(policy.model)
  optimizer = optim.RMSprop(
    model.parameters(), lr=0.01, alpha=0.99, eps=1e-8, weight_decay=0
  )
  rb = ReplayBuffer(replay_buffer_len, batch_size)
  rewards_sum = 0
  rewards = []
  moving_avg = []
  policy.EPS = eps_start
  warmup_steps = batch_size * 2
  beta_increment = (1 - beta) / (total_episodes * 500)

  plt.ion()
  _, ax = plt.subplots()
  (line,) = ax.plot([], [])
  step_count = 0

  while len(rewards) <= total_episodes:
    action_idx = policy.get_action(torch.tensor([obs]))
    action_idx = action_idx[0].item()

    obs_p, reward, terminated, truncated, _ = env.step(action_idx)
    rewards_sum += reward
    step_count += 1

    experience = (obs, action_idx, reward, obs_p, terminated)
    rb.insert(experience)

    if terminated or truncated:
      obs, _ = env.reset()
      if terminated:
        print(f"Terminated! Rewards: {rewards_sum}, Epsilon: {policy.EPS:.3f}")
      else:
        print(f"Truncated! Rewards: {rewards_sum}, Epsilon: {policy.EPS:.3f}")

      moving_avg.append(
        (rewards_sum + sum(rewards[-min(window_len - 1, len(rewards)) :]))
        / min(window_len, len(rewards) + 1)
      )
      rewards.append(rewards_sum)
      rewards_sum = 0

      # Decay epsilon after each episode
      policy.EPS = max(eps_end, policy.EPS * eps_decay)

      # Update plot
      line.set_xdata(range(len(moving_avg)))
      line.set_ydata(moving_avg)
      ax.relim()
      ax.autoscale_view()
      plt.pause(0.01)
    else:
      obs = obs_p

    # Skip training until replay buffer has enough samples
    if step_count < warmup_steps:
      continue

    # Train every step after warmup
    optimizer.zero_grad()
    batches, indices, weights = rb.sample(beta)
    batch_obs = torch.tensor([i[0] for i in batches])
    batch_action_idx = torch.tensor([i[1] for i in batches])
    batch_reward = torch.tensor([i[2] for i in batches], dtype=torch.float32)
    batch_obs_p = torch.tensor([i[3] for i in batches])
    batch_terminated = torch.tensor([i[4] for i in batches], dtype=torch.float32)

    action_values = policy.model(batch_obs)
    action_values = torch.gather(
      action_values, dim=1, index=batch_action_idx.unsqueeze(1)
    ).squeeze(1)

    next_online_action_values = policy.model(batch_obs_p).detach()
    next_max_action = (
      torch.gather(
        target_model(batch_obs_p),
        dim=1,
        index=torch.argmax(next_online_action_values, dim=1).unsqueeze(1),
      )
      .squeeze(1)
      .detach()
    )
    targets = batch_reward + gamma * (1 - batch_terminated) * next_max_action
    td_err = action_values - targets

    loss = ((weights * td_err).pow(2)).mul(0.5).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.model.parameters(), max_norm=1.0)
    optimizer.step()

    # exponential update
    with torch.no_grad():
      td_err_detached = td_err.detach()
      for i in range(len(indices)):
        rb.update(indices[i], td_err_detached[i].abs())

      for target, online in zip(target_model.parameters(), policy.model.parameters()):
        target.data.copy_(tau * target.data + (1 - tau) * online.data)

      beta = min(1, beta + beta_increment)

  print("Training finished!")

  plt.ioff()
  plt.show()


def simulate(policy):
  env = gym.make("CartPole-v1", render_mode="human")
  obs, _ = env.reset()
  rewards = 0

  while True:
    action_idx = policy.get_action(torch.tensor([obs]), True)

    action_idx = action_idx[0].item()

    obs_p, reward, terminated, truncated, _ = env.step(action_idx)

    rewards += reward

    if terminated or truncated:
      obs, _ = env.reset()
      break
    else:
      obs = obs_p

  print("Total rewards: ", rewards)


gamma = 0.99
state_size = 4
action_size = 2
batch_size = 64
model = DuelingQNetwork(state_size, action_size)
policy = Policy(model, state_size, action_size)
vis_window_len = 30
replay_buffer_len = 50000
target_update_freq = 500
total_episodes = 600
beta = 0.4

train(
  policy,
  batch_size,
  gamma,
  total_episodes,
  replay_buffer_len,
  vis_window_len,
  target_update_freq,
  beta=beta,
)

simulate(policy)
