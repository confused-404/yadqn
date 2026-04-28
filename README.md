# YADQN (Yet Another DQN)

Deep Q-Network implemented from scratch in NumPy to explore reinforcement learning dynamics without relying on ML frameworks.

## Overview

This project trains a DQN agent on the CartPole environment using only NumPy. The goal was to understand the core mechanics of reinforcement learning—Q-value estimation, stability issues, and training dynamics—by implementing everything manually.

Environment: https://gymnasium.farama.org/environments/classic_control/cart_pole/

## Features

* Double DQN for more stable Q-value estimation
* Huber loss for robustness to outliers
* Experience replay buffer
* Target network updates
* Training logs with performance visualization

## Results

The agent consistently learns to balance the pole and reaches the maximum episode length (500 steps).

 ![training_progress](https://github.com/confused-404/yadqn/blob/main/training_progress.png?raw=true)

## Observations

* The model undergoes an initial exploration phase (~100–150 episodes)
* It quickly converges to near-optimal performance
* After convergence, Q-values can become unstable and explode, causing performance collapse
* This highlights the sensitivity of DQN to hyperparameters and training stability

## Technical Highlights

* Implemented full forward/backward pass manually in NumPy
* Built replay buffer and batch sampling system
* Managed target network synchronization for stability
* Investigated Q-value divergence and training instability

## Running

```bash id="yadqn-run"
pip install -r requirements.txt
python train.py
```

## Why this project

Most RL implementations rely on frameworks like PyTorch or TensorFlow. This project focuses on understanding the underlying algorithms by removing that abstraction and implementing DQN from first principles.

## Limitations / Future Work

* Improve stability to prevent Q-value explosion
* Add prioritized experience replay
* Extend to more complex environments
