import copy
import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.optim as optim

class QNetwork(torch.nn.Module):
  def __init__(self, state_size, action_size):
    super(QNetwork, self).__init__()
    self.fc1 = torch.nn.Linear(state_size, 512)
    self.fc2 = torch.nn.Linear(512, 512)
    self.fc3 = torch.nn.Linear(512, action_size)

  def forward(self, x):
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = self.fc3(x)
    return x

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
    self.cnt = 0
    self.rb = [tuple() for _ in range(length)]
  
  def insert(self, experience):
    self.rb[self.cnt % self.length] = experience
    self.cnt += 1
  
  def sample(self):
    return [self.rb[i] for i in np.random.randint(0, min(self.cnt, self.length), self.batch_size)]


def train(policy, batch_size, gamma, total_episodes, replay_buffer_len, window_len, target_update_freq, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
  env = gym.make("CartPole-v1", render_mode=None)
  obs, _= env.reset()

  target_model = copy.deepcopy(policy.model)
  criterion = torch.nn.MSELoss()
  optimizer = optim.RMSprop(model.parameters(),
                            lr=0.01,
                            alpha=0.99,     
                            eps=1e-8,
                            weight_decay=0)
  rb = ReplayBuffer(replay_buffer_len, batch_size)
  rewards_sum = 0
  rewards = []
  moving_avg = []
  policy.EPS = eps_start
  warmup_steps = batch_size * 2

  plt.ion()  
  _, ax = plt.subplots()
  line, = ax.plot([], []) 
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

      moving_avg.append((rewards_sum + sum(rewards[-min(window_len-1, len(rewards)):])) / min(window_len, len(rewards)+1))
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
    batches = rb.sample()
    batch_obs = torch.tensor([i[0] for i in batches])
    batch_action_idx = torch.tensor([i[1] for i in batches])
    batch_reward = torch.tensor([i[2] for i in batches], dtype=torch.float32)
    batch_obs_p = torch.tensor([i[3] for i in batches])
    batch_terminated = torch.tensor([i[4] for i in batches], dtype=torch.float32)

    action_values = policy.model(batch_obs)
    action_values = torch.gather(action_values, dim=1, index=batch_action_idx.unsqueeze(1)).squeeze(1)

    next_max_action = torch.max(target_model(batch_obs_p), dim=-1).values.detach()
    targets = batch_reward + gamma * (1 - batch_terminated) * next_max_action
    
    loss = criterion(action_values, targets)
    loss.backward()
    optimizer.step()

    # Update target network every N steps
    if step_count % (target_update_freq) == 0:
      target_model = copy.deepcopy(policy.model)
      print(f"Target network updated at step {step_count}")      

  print("Training finished!")

  plt.ioff()
  plt.show()

def simulate(policy):
  env = gym.make("CartPole-v1", render_mode='human')
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
model = QNetwork(state_size, action_size)
policy = Policy(model, state_size, action_size)
vis_window_len = 30
replay_buffer_len = 50000
target_update_freq = 500
total_episodes=600

train(policy, batch_size, gamma, total_episodes, replay_buffer_len, vis_window_len, target_update_freq)

simulate(policy)