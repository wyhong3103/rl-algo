# rl-algo

This repository includes a collection of Reinforcement Learning (RL) algorithms I have implemented, ranging from traditional tabular methods to modern function approximation techniques, to solve classic control tasks using environments from [Farama Gymnasium](https://gymnasium.farama.org/).

I implemented these algorithms as part of my learning process while reading [Reinforcement Learning: An Introduction](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf) by Sutton and Barto, and [Grokking Deep Reinforcement Learning](https://www.manning.com/books/grokking-deep-reinforcement-learning) by Miguel Morales. The former lays the foundation of RL, focusing primarily on tabular methods and linear function approximation, while the latter delves into Deep RL, covering the most famous methods in each Deep RL paradigm.

If you are interested in RL, I highly recommend both books! However, since the latter doesn't go deeply into the mathematics, I encourage reading original research papers (and using AI) alongside it.

I'm no expert on this subject, I'm learning too. Just sharing this in case it helps anyone else who’s figuring it out as well.

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

> The implementation style may be inconsistent across methods because I was constantly restructuring the code, as I felt the previous versions weren't good enough. I'll admit I got a bit lazy in some parts, so certain sections aren't as clean as they could be. Additionally, some implementations may not perfectly match the original papers. If you notice any discrepancies, please feel free to open an issue.

## If you're also learning RL...

I recommend these resources:

1. [Reinforcement Learning: An Introduction](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf). The RL bible, a very good book. This should be your first read.
2. [Grokking Deep Reinforcement Learning](https://www.manning.com/books/grokking-deep-reinforcement-learning). Covers modern Deep RL methods. It's quite shallow, but good enough to build intuition.
3. [CMU: 2018 Fall: 10-703 Deep Reinforcement Learning](https://www.youtube.com/playlist?list=PLpIxOj-HnDsNfvOwRKLsUobmnF2J1l5oV). A good lecture series. I've watched a few of the lectures and think it's solid.
4. Papers. Some methods (such as TRPO and PPO) deserve a careful read!

Here’s a good piece of advice (word for word):

> Mist important tip: Actually do implement things. People allways THINK they understand RL after reading some papers, until they actually implement and learn they actually don't really understand things. Implementing yourself is the only way to actually understand. Otherwise you are just like the drunk people screaming at their TV blasting football and thinking they could do that Imho. [[source]](https://www.reddit.com/r/MachineLearning/comments/1779tv0/comment/k4rrdxy/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button)
