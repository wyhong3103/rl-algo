from __future__ import annotations
import matplotlib.pyplot as plt
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
import numpy as np

class SimpleEnv(MiniGridEnv):
  def __init__(
    self,
    size=10,
    agent_start_pos=(1, 1),
    agent_start_dir=0,
    max_steps: int | None = None,
    agent_view_size = 3,
    **kwargs,
  ):
    self.agent_start_pos = agent_start_pos
    self.agent_start_dir = agent_start_dir

    mission_space = MissionSpace(mission_func=self._gen_mission)

    if max_steps is None:
        max_steps = 4 * size**2

    super().__init__(
        mission_space=mission_space,
        grid_size=size,
        see_through_walls=True,
        max_steps=max_steps,
        agent_view_size=agent_view_size,
        **kwargs,
    )

  @staticmethod
  def _gen_mission():
    return "grand mission"

  def _gen_grid(self, width, height):
    self.grid = Grid(width, height)

    self.grid.wall_rect(0, 0, width, height)

    for i in range(0, height):
        if i != height // 2:
          self.grid.set(width // 3, i, Wall())

    for i in range(0, height):
        if i != height // 3:
          self.grid.set((width * 2) // 3, i, Wall())

    self.put_obj(Goal(), width - 2, height - 2)

    if self.agent_start_pos is not None:
        self.agent_pos = self.agent_start_pos
        self.agent_dir = self.agent_start_dir
    else:
        self.place_agent()

    self.mission = "grand mission"

  def step(self, action):
    # Call the original step function
    obs, reward, done, truncated, info = super().step(action)
    
    # Subtract 1 for every step
    if not done:
      reward = -1
    else:
      reward = 10000

    return obs, reward, done, truncated, info

class Sarsa:
  def __init__(self, r, c):
    self.q = np.zeros((r, c, 4, 3))
    self.eps = 1e-3
    self.alpha = 1e-1
    self.gamma = 0.5
    
  def get_action(self, r, c, d): 
    p = np.random.rand()
    return np.argmax(self.q[r][c][d]) if p > self.eps else np.random.randint(0, 3)
  
  def get_max_action(self, r, c, d):
    return np.argmax(self.q[r][c][d])
  
  def get_algo_type(self):
    return "sarsa"

  def update(self, reward, r, c, d, a, n_r, n_c, n_d, n_a):
    self.q[r][c][d][a] = (
      self.q[r][c][d][a] + 
      self.alpha * (reward + self.gamma * self.q[n_r][n_c][n_d][n_a] - self.q[r][c][d][a])
    )

class QLearning:
  def __init__(self, r, c):
    self.q = np.zeros((r, c, 4, 3))
    self.eps = 1e-3
    self.alpha = 1e-1
    self.gamma = 0.5
    
  def get_action(self, r, c, d): 
    p = np.random.rand()
    return np.argmax(self.q[r][c][d]) if p > self.eps else np.random.randint(0, 3)
  
  def get_max_action(self, r, c, d):
    return np.argmax(self.q[r][c][d])
  
  def get_algo_type(self):
    return "qlearning"
  
  def update(self, reward, r, c, d, a, n_r, n_c, n_d):
    self.q[r][c][d][a] = (
      self.q[r][c][d][a] + 
      self.alpha * (reward + self.gamma * self.q[n_r][n_c][n_d][self.get_max_action(n_r, n_c, n_d)] - self.q[r][c][d][a])
    )

class SarsaWithET:
  def __init__(self, r, c):
    self.q = np.zeros((r, c, 4, 3))
    self.e = np.zeros((r, c, 4, 3))
    self.eps = 1e-3
    self.alpha = 1e-1
    self.lmbda = 0.5
    self.gamma = 0.5
    
  def get_action(self, r, c, d): 
    p = np.random.rand()
    return np.argmax(self.q[r][c][d]) if p > self.eps else np.random.randint(0, 3)
  
  def get_max_action(self, r, c, d):
    return np.argmax(self.q[r][c][d])
  
  def get_algo_type(self):
    return "sarsa_with_eligibility_trace"

  def update(self, reward, r, c, d, a, n_r, n_c, n_d, n_a):
    delta = reward + self.gamma * self.q[n_r][n_c][n_d][n_a] - self.q[r][c][d][a]
    self.e[r][c][d][a] += 1

    self.q += self.alpha * delta * self.e
    self.e = self.gamma * self.lmbda * self.e
  
class WatkinQET:
  def __init__(self, r, c):
    self.q = np.zeros((r, c, 4, 3))
    self.e = np.zeros((r, c, 4, 3))
    self.eps = 1e-3
    self.alpha = 1e-1
    self.lmbda = 0.5
    self.gamma = 0.5
    
  def get_action(self, r, c, d): 
    p = np.random.rand()
    return np.argmax(self.q[r][c][d]) if p > self.eps else np.random.randint(0, 3)
  
  def get_max_action(self, r, c, d):
    return np.argmax(self.q[r][c][d])
  
  def get_algo_type(self):
    return "watkins_q_with_eligibility_trace"

  def update(self, reward, r, c, d, a, n_r, n_c, n_d, n_a):
    a_star = self.get_max_action(n_r, n_c, n_d)
    delta = reward + self.gamma * self.q[n_r][n_c][n_d][a_star] - self.q[r][c][d][a]
    self.e[r][c][d][a] += 1

    self.q += self.alpha * delta * self.e
    if n_a == a_star:
      self.e = self.gamma * self.lmbda * self.e
    else:
      self.e *= 0

def run_episode(env, policy, train=True):
    obs, _ = env.reset()
    is_wall = obs['image'][1][1][0] == 2
    r, c, d = 0, 0, 0
    a = policy.get_action(r, c, d)
    algo = policy.get_algo_type()
    cnt = 0
    for _ in range(10000):
      obs, reward, done, truncated, _ = env.step(a)

      print(f"Action: {a}, Reward: {reward}, Done: {done}, Truncated: {truncated}")

      cnt += 1
      if done or truncated:
        if done:
          print(f"Took {cnt} steps")
          return cnt
        else:
          print("Truncated")
          return -1

      n_r, n_c, n_d = r, c, d
      if a == 0:
        n_d = (n_d - 1) % 4
      elif a == 1:
        n_d = (n_d + 1) % 4
      else:
        if not is_wall:
          if n_d == 0:
            n_c += 1
          elif n_d == 1:
            n_r += 1
          elif n_d == 2:
            n_c -= 1
          else:
            n_r -= 1

      if train:
        n_a = policy.get_action(n_r, n_c, n_d)
      else:
        n_a = policy.get_max_action(n_r, n_c, n_d)

      if train:
        if algo in ["watkins_q_with_eligibility_trace", "sarsa_with_eligibility_trace", "sarsa"]:
          policy.update(reward, r, c, d, a, n_r, n_c, n_d, n_a)
        else:
          policy.update(reward, r, c, d, a, n_r, n_c, n_d)

      r, c, d, a = n_r, n_c, n_d, n_a
      is_wall = obs['image'][1][1][0] == 2

def main():
  size = 20
  env = SimpleEnv(render_mode=None, size=size)

  # policy = QLearning(size, size)
  # policy = Sarsa(size, size)
  # policy = SarsaWithET(size, size)
  policy = WatkinQET(size, size)

  steps = []
  for _ in range(3000):
    total_steps = run_episode(env, policy)
    steps.append(total_steps)
  
  env = SimpleEnv(render_mode='human', size=size)
  run_episode(env, policy)

  plt.plot(range(1, len(steps)+1), steps)
  plt.title(policy.get_algo_type())
  plt.show()

if __name__ == "__main__":
  main()