# rl-algo

This repository includes a collection of Reinforcement Learning (RL) algorithms I have implemented, ranging from traditional tabular methods to modern function approximation techniques, to solve classic control tasks using environments from [Farama Gymnasium](https://gymnasium.farama.org/).

I implemented these algorithms as part of my learning process while reading [Reinforcement Learning: An Introduction](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf) by Sutton and Barto, and [Grokking Deep Reinforcement Learning](https://www.manning.com/books/grokking-deep-reinforcement-learning) by Miguel Morales. The former lays the foundation of RL, focusing primarily on tabular methods and linear function approximation, while the latter delves into Deep RL, covering the most famous methods in each Deep RL paradigm.

If you are interested in RL, I highly recommend both books! However, since the latter doesn't go deeply into the mathematics, I encourage reading original research papers (and using AI) alongside them.

## Algorithms

Tabular Methods

1. [Multi Armed Bandit](/tabular/multi-bandit.py)
2. [Monte Carlo](/tabular/monte-carlo.py)
3. [Temporal Difference Learning](/tabular/temporal-difference.py)

Deep RL Methods

1. [Neural Fitted Q Learning](/value-based/nfq.py)
2. [Deep Q Network](/value-based/dqn.py)
3. [Double Deep Q Network](/value-based/ddqn.py)
4. [Dueling Double Q Network](/value-based/dueling-ddqn.py)
5. [REINFORCE](/actor-critic/reinforce.py)
6. [Asynchronous Advantage Actor Critic](/actor-critic/a3c-gae.py)
7. [Deep Deterministic Policy Gradient](/actor-critic/ddpg.py)
8. [Soft Actor-Critic](/actor-critic/sac.py)
9. [Proximal Policy Optimization](/actor-critic/ppo.py)

> Some implementations might not perfectly match the original papers. If you find any discrepancies, please feel free to open an issue.
