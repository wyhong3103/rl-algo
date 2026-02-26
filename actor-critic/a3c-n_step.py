import gymnasium as gym
import torch
from torch import nn
import matplotlib.pyplot as plt
import multiprocessing


class ValueDNN(nn.Module):
  def __init__(self, input_dim, hidden_dims, output_dim):
    super(ValueDNN, self).__init__()
    self.input_layer = nn.Linear(input_dim, hidden_dims[0])
    self.hidden_layers = nn.ModuleList()
    for i in range(len(hidden_dims) - 1):
      self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
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
    for i in range(len(hidden_dims) - 1):
      self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
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


class SharedAdam(torch.optim.Adam):
  def __init__(
    self,
    params,
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0,
    amsgrad=False,
  ):
    super(SharedAdam, self).__init__(
      params,
      lr=lr,
      betas=betas,
      eps=eps,
      weight_decay=weight_decay,
      amsgrad=amsgrad,
    )

    for group in self.param_groups:
      for p in group["params"]:
        state = self.state[p]
        state["step"] = torch.zeros(1)
        state["shared_step"] = torch.zeros(1).share_memory_()
        state["exp_avg"] = torch.zeros_like(p.data).share_memory_()
        state["exp_avg_sq"] = torch.zeros_like(p.data).share_memory_()
        if weight_decay:
          state["weight_decay"] = torch.zeros_like(p.data).share_memory_()
        if amsgrad:
          state["max_exp_avg_sq"] = torch.zeros_like(p.data).share_memory_()

  def step(self, closure=None):
    for group in self.param_groups:
      for p in group["params"]:
        if p.grad is None:
          continue
        self.state[p]["step"] = self.state[p]["shared_step"]
        self.state[p]["shared_step"] += 1
    super().step(closure)


class A3C:
  def __init__(self):
    self.global_policy_model = PolicyDNN(4, [128, 128], 2).share_memory()
    self.global_value_model = ValueDNN(4, [128, 128], 1).share_memory()
    self.global_policy_optimizer = SharedAdam(
      self.global_policy_model.parameters(), lr=1e-3
    )
    self.global_value_optimizer = SharedAdam(
      self.global_value_model.parameters(), lr=1e-3
    )
    self.gamma = 0.99
    self.total_episodes = 500
    self.n = 100
    self.n_workers = 3

  def optimize(
    self, local_policy_model, local_value_model, rewards, logpas, values, entropys
  ):
    logpas = torch.stack(logpas).view(-1)
    values = torch.stack(values).view(-1)
    entropys = torch.stack(entropys).view(-1)

    returns = []
    g_sum = rewards[-1]
    for r in reversed(rewards[:-1]):
      g_sum = r + self.gamma * g_sum
      returns.insert(0, g_sum)

    returns = torch.tensor(returns, dtype=torch.float32)
    value_errors = returns - values

    policy_loss = -(value_errors.detach() * logpas).mean() - 0.01 * entropys.mean()

    self.global_policy_optimizer.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(local_policy_model.parameters(), max_norm=1.0)
    for param, shared_param in zip(
      local_policy_model.parameters(), self.global_policy_model.parameters()
    ):
      if shared_param.grad is None:
        shared_param._grad = param.grad
    self.global_policy_optimizer.step()
    local_policy_model.load_state_dict(self.global_policy_model.state_dict())

    value_loss = value_errors.pow(2).mean()
    self.global_value_optimizer.zero_grad()
    value_loss.backward()
    torch.nn.utils.clip_grad_norm_(local_value_model.parameters(), max_norm=1.0)
    for param, shared_param in zip(
      local_value_model.parameters(), self.global_value_model.parameters()
    ):
      if shared_param.grad is None:
        shared_param._grad = param.grad
    self.global_value_optimizer.step()
    local_value_model.load_state_dict(self.global_value_model.state_dict())

  def worker_train(self, rank):
    env = gym.make("CartPole-v1")
    seed = 42 + rank
    local_policy_model = PolicyDNN(4, [128, 128], 2)
    local_policy_model.load_state_dict(self.global_policy_model.state_dict())
    local_value_model = ValueDNN(4, [128, 128], 1)
    local_value_model.load_state_dict(self.global_value_model.state_dict())

    if rank == 0:
      plt.ion()
      _, ax = plt.subplots()
      (line,) = ax.plot([], [])
      window_len = 30
      returns = []
      moving_avg = []

    for episode in range(self.total_episodes):
      obs, _ = env.reset(seed=seed)
      global_rewards = []

      terminated = False
      current_step = 0
      rewards = []
      logpas = []
      values = []
      entropys = []

      while not terminated:
        action_idx, logpa, entropy = local_policy_model.full_pass(torch.tensor(obs))
        value = local_value_model(torch.tensor(obs))
        current_step += 1

        obs_p, reward, terminated, truncated, _ = env.step(action_idx)

        global_rewards.append(reward)
        rewards.append(reward)
        logpas.append(logpa)
        values.append(value)
        entropys.append(entropy)

        if not terminated:
          obs = obs_p

        if current_step == self.n or truncated:
          value = local_value_model(torch.tensor(obs))
          rewards.append(value.item())
        elif terminated:
          rewards.append(0)

        if truncated:
          terminated = True

        if current_step == self.n or terminated:
          self.optimize(
            local_policy_model,
            local_value_model,
            rewards,
            logpas,
            values,
            entropys,
          )
          current_step = 0
          rewards = []
          logpas = []
          values = []
          entropys = []

      if rank == 0:
        print(f"Episode {episode}: {sum(global_rewards)} rewards")

        moving_avg.append(
          (sum(global_rewards) + sum(returns[-min(window_len - 1, len(returns)) :]))
          / min(window_len, len(returns) + 1)
        )
        line.set_xdata(range(len(moving_avg)))
        line.set_ydata(moving_avg)
        ax.relim()
        ax.autoscale_view()
        plt.pause(0.01)

  def train(self):
    processes = []

    for i in range(self.n_workers):
      p = multiprocessing.Process(target=self.worker_train, args=(i,))
      processes.append(p)
      p.start()

    print(f"--- All {self.n_workers} workers have started running. ---")

    for p in processes:
      p.join()

    print("--- All workers have completed training. ---")

  def simulate(self):
    env = gym.make("CartPole-v1", render_mode="human")
    obs, _ = env.reset()
    rewards = 0
    terminated = False

    while not terminated:
      action_idx = self.global_policy_model.get_action(torch.tensor(obs), True)

      action_idx = action_idx

      obs_p, reward, terminated, truncated, _ = env.step(action_idx)

      rewards += reward

      obs = obs_p

      if truncated:
        terminated = True

    print("Total rewards: ", rewards)


if __name__ == "__main__":
  rl = A3C()
  rl.train()
  rl.simulate()
