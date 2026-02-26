import numpy as np

MN_MEAN = 1
MX_MEAN = 1000
MN_VAR = 1
MX_VAR = 100
N = 5
EPS = 1e-1
ALPHA = 1e-2
T = 10000
means = [1, 4, 6, 7, 10]  # np.random.randint(MN_MEAN, MX_MEAN, size=(N))
variances = [5, 2, 5, 5, 2]  # np.random.randint(MN_VAR, MX_VAR, size=(N))
average = [0] * N
cnt = [0] * N

for t in range(T):
  action = np.argmax(average)
  p = np.random.rand()
  if p <= EPS:
    action = np.random.randint(0, N)
  reward = np.random.normal(loc=means[action], scale=variances[action])
  average[action] = average[action] + (1 / (cnt[action] + 1)) * (
    reward - average[action]
  )
  cnt[action] += 1
  EPS = (1 - ALPHA) * EPS
  print(average)
  print(action, reward)
