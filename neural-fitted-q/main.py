# Not entirely accurate since I didn't train the model on the same batch repeatedly for K times
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

    p = np.random.rand()

    action_idx = torch.argmax(a_p).item()

    if not get_max and p <= self.EPS:
      action_idx = np.random.randint(0, self.action_size)  

    return action_idx, a_p[action_idx]

def train(policy, optimizer, batch_size, criterion, gamma, total_episodes, window_len):
  env = gym.make("CartPole-v1", render_mode=None)
  obs, info = env.reset()

  plt.ion()  
  _, ax = plt.subplots()
  line, = ax.plot([], []) 

  rewards_sum = 0
  rewards = []

  while len(rewards) <= total_episodes:
    optimizer.zero_grad()
    action_values = torch.zeros((batch_size))
    targets = torch.zeros((batch_size))

    for i in range(batch_size):
      action_idx, action_value = policy.get_action(torch.tensor(obs))

      obs_p, reward, terminated, truncated, info = env.step(action_idx)

      rewards_sum += reward

      next_max_action = 0 if terminated and not truncated else policy.get_action(torch.tensor(obs_p), True)[1].detach()

      target = reward + gamma * next_max_action

      action_values[i] = action_value
      targets[i] = target

      if terminated or truncated:
        obs, info = env.reset()
        if terminated:
          print(f"Teminated! Rewards", rewards_sum)
        else:
          print(f"Truncated! Rewards", rewards_sum)

        rewards.append((rewards_sum + sum(rewards[-min(window_len-1, len(rewards)):])) / min(window_len, len(rewards)+1))
        rewards_sum = 0
      else:
        obs = obs_p
    
    loss = criterion(action_values, targets)
    loss.backward()
    optimizer.step()

    line.set_xdata(range(len(rewards)))
    line.set_ydata(rewards)
    ax.relim()           
    ax.autoscale_view()  
    plt.pause(0.01)      

  print("Training finished!")

  plt.ioff()
  plt.show()

def simulate(policy):
  env = gym.make("CartPole-v1", render_mode='human')
  obs, info = env.reset()
  rewards = 0

  while True:
    action_idx, action_value = policy.get_action(torch.tensor(obs), True)

    obs_p, reward, terminated, truncated, info = env.step(action_idx)

    rewards += reward

    if terminated or truncated:
      obs, info = env.reset()
      break
    else:
      obs = obs_p
    
  print("Total rewards: ", rewards)

gamma = 0.99
state_size = 4
action_size = 2
batch_size = 1024
criterion = torch.nn.MSELoss()
model = QNetwork(state_size, action_size)
policy = Policy(model, state_size, action_size)
optimizer = optim.RMSprop(model.parameters(),
                          lr=0.01,
                          alpha=0.99,     
                          eps=1e-8,
                          weight_decay=0)
window_len = 30
total_episodes=5000

train(policy, optimizer, batch_size, criterion, gamma, total_episodes, window_len)

simulate(policy)