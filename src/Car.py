import State;

class Car:
    # Creates the car
    def _init__(self, action, reward, epsilon_start, epsilon_end, epsilon_decay):
        self.action = action
        self.reward = reward
        self.state = State(epsilon_start, epsilon_end, epsilon_decay)

    # Choose forward, left, right, or nothing using State
    def chooseAction(self):

    # Do Action w/ action variable
    def doAction(self):

    # Updates the stats of State and ReplayMemory
    def update(self):
