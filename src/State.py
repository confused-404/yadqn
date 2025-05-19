class State:
    def __init__(self, epsilon_start, epsilon_end, epsilon_decay):
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay