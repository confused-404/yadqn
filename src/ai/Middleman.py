from ai.DQN import DQN
from ai.ReplayMemory import ReplayMemory, Transition
import ai.parameters as params
import math
import random
import numpy as np
import gymnasium as gym
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

class Middleman:
    def __init__(self, env: gym.Env):
        self.env = env
        state, info = self.env.reset()
        
        self.n_observations = len(state)
        self.n_actions = self.env.action_space.n # only for discrete action spaces
        
        self.policy_net = DQN(self.n_observations, [128, 128], self.n_actions)
        self.target_net = DQN(self.n_observations, [128, 128], self.n_actions)
        self.target_net.load(self.policy_net)

        self.memory = ReplayMemory(params.REPLAY_BUFFER_SIZE)
        
        self.steps_done = 0
        
    def select_action(self, state):
        sample = random.random()
        eps_threshold = params.EPS_END + (params.EPS_START - params.EPS_END) * \
            math.exp(-1. * self.steps_done / params.EPS_DECAY)
        self.steps_done += 1
        
        if sample > eps_threshold:
            q_values = self.policy_net.forward(state)
            return np.array([[np.argmax(q_values)]])  # pick best action
        else:
            return np.array([[self.env.action_space.sample()]])
        
    def huber_loss(self, y_true, y_pred, delta=1.0):
        error = y_true - y_pred
        abs_error = np.abs(error)
        quadratic = np.minimum(abs_error, delta)
        linear = abs_error - quadratic
        loss = 0.5 * quadratic**2 + delta * linear
        return loss.mean(), np.where(abs_error <= delta, -error, -delta * np.sign(error))
    
    def optimize_model(self):
        if len(self.memory) < params.BATCH_SIZE:
            return None, None
        
        transitions = self.memory.sample(batch_size=params.BATCH_SIZE)
        
        # transpose the batch (batch-array of transitions -> transition of batch-arrays)
        """
        Transition(
            state=(s0, s1, s2),
            action=(a0, a1, a2),
            next_state=(s1, s2, s3),
            reward=(r0, r1, r2),
            done=(d0, d1, d2)
        )
        """
        batch = Transition(*zip(*transitions))
        
        non_final_mask = np.array([s is not None for s in batch.next_state], dtype=bool)
        non_final_next_states_list = [s for s in batch.next_state if s is not None]
        state_dim = batch.state[0].shape[1]
        
        if non_final_next_states_list:
            non_final_next_states = np.concatenate(non_final_next_states_list, axis=0)
        else:
            non_final_next_states = np.empty((0, state_dim))
        
        state_batch = np.concatenate(batch.state)
        action_batch = np.concatenate(batch.action)
        reward_batch = np.concatenate(batch.reward)
        
        # shape (params.BATCH_SIZE, self.n_actions)
        batch_indices = np.arange(action_batch.shape[0])
        action_indices = action_batch.flatten()
        full_q_values = self.policy_net.forward(state_batch)
        state_action_values = full_q_values[batch_indices, action_indices][:, np.newaxis]
        
        next_state_values = np.zeros((params.BATCH_SIZE, 1))
        next_state_values[non_final_mask] = np.max(self.target_net.forward(non_final_next_states), axis=1, keepdims=True)

        expected_state_action_values = (next_state_values * params.GAMMA) + reward_batch
        
        # huber loss (mean absolute error when error is large to avoid crazy loss with mean squared error)
        loss_value, grad_loss_output = self.huber_loss(expected_state_action_values, state_action_values)
        
        # backprop the gradient through nn
        grad_full = np.zeros_like(full_q_values)
        grad_full[batch_indices, action_indices] = grad_loss_output.flatten()
        self.policy_net.backward(grad_full)
        
        # clip gradients
        self.policy_net.clip_gradients(max_val=100)
        
        # update policy net
        self.policy_net.update(params.LR)
        
        return [loss_value, np.mean(full_q_values)]

    # def copy_policy_to_target(self):
    #     self.target_net.load(self.policy_net)
        
    def train(self, render=False, log_progress=False):
        best_duration = 0
        steps_per_episode = []

        episode_range = trange(params.NUM_EPISODES, desc="Training", leave=True) if log_progress else range(params.NUM_EPISODES)

        for i_episode in episode_range:
            state, info = self.env.reset()
            state = np.expand_dims(state, axis=0)
            total_steps = 0

            for t in range(params.MAX_TIMESTEPS):
                action = self.select_action(state)
                action_value = int(action[0, 0])

                next_obs, reward, terminated, truncated, _ = self.env.step(action_value)
                if render:
                    self.env.render()
                done = terminated or truncated

                reward = np.array([[reward]])
                next_state = None if terminated else np.expand_dims(next_obs, axis=0)

                self.memory.push(state, action, next_state, reward)

                state = next_state

                loss_value, mean_q = self.optimize_model()

                # hard target net sync every 100 steps
                if self.steps_done % 100 == 0:
                    self.target_net.load(self.policy_net)
                # else:
                #     for target_layer, policy_layer in zip(self.target_net.layers, self.policy_net.layers):
                #         target_layer.weights = params.TAU * policy_layer.weights + (1 - params.TAU) * target_layer.weights
                #         target_layer.biases = params.TAU * policy_layer.biases + (1 - params.TAU) * target_layer.biases

                total_steps += 1
                if done:
                    break
            
            best_duration = max(best_duration, total_steps)
            steps_per_episode.append(total_steps)

            if log_progress:
                episode_range.set_description(f"Ep {i_episode + 1}/{params.NUM_EPISODES}")
                episode_range.set_postfix(Steps=total_steps, Best=best_duration)
                
            if log_progress and loss_value is not None and i_episode % 10 == 0:
                tqdm.write(f"Ep {i_episode} | loss: {loss_value:.4f} | mean_q: {mean_q:.2f}")

        # Save plot
        if log_progress:
            smoothed = uniform_filter1d(steps_per_episode, size=20) # moving average line
            plt.figure(figsize=(10, 5))
            plt.plot(steps_per_episode, label="Steps per Episode")
            plt.plot(smoothed, label="Smoothed")
            plt.xlabel("Episode")
            plt.ylabel("Steps")
            plt.title("Training Progress")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig("training_progress.png")
            plt.close()