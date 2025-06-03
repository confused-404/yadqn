from ai.DQN import DQN
from ai.ReplayMemory import ReplayMemory
import ai.parameters as params
import math
import random
import numpy as np

class Middleman:
    def __init__(self, env):
        self.env = env
        state, info = info.reset()
        
        self.n_actions = self.env.action_space.n
        self.n_observations = len(state)
        
        self.policy_net = DQN(self.n_observations, [], self.n_actions)
        self.target_net = DQN(self.n_observations, [], self.n_actions)
        self.target_net.load(self.policy_net)
        
        self.memory = ReplayMemory(params.REPLAY_BUFFER_SIZE)
        
        self.steps_done = 0
        
    def select_action(self, state):
        sample = random.random()
        eps_threshold = params.EPS_END + (params.EPS_START - params.EPS_END) * \
            math.exp(-1. * self.steps_done / params.EPS_DECAY)
        self.steps_done += 1
        
        if sample > eps_threshold:
            return self.policy_net.forward(state)
        else:
            return np.array([[self.env.action_space.sample()]])