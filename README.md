# yadqn

A very simple DQN to learn more about reinforcement learning in premade gymnasium environments.

Name is like yaml -> yet another dqn

## Environment

https://gymnasium.farama.org/environments/classic_control/cart_pole/

## Notes

Followed https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html to learn

Didn't use pytorch, wrote everything from scratch with NumPy

## Issues

- The model experiences an exploration phase for the first 100-150 episodes, and then rapidly reaches max steps (500)
  - However, after ~10-20 episodes at max steps, the model becomes overconfident and Q values explode, causing it to reset down to 10 steps