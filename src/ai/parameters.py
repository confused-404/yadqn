# FROM PYTORCH

BATCH_SIZE = 1000 # BATCH_SIZE is the number of transitions sampled from the replay buffer
GAMMA = 0.99 # GAMMA is the discount factor as mentioned in the previous section
EPS_START = 0.9 # EPS_START is the starting value of epsilon
EPS_END = 0.05 # EPS_END is the final value of epsilon
EPS_DECAY = 2000 # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
TAU = 0.005 # TAU is the update rate of the target network
LR = 5e-5 # LR is the learning rate of the ``AdamW`` optimizer
REPLAY_BUFFER_SIZE = int(10e4)
NUM_EPISODES = 500
MAX_TIMESTEPS = 500