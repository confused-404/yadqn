import State;

class Car:
    # Create the car AI
    def _init__(self, action, reward, epsilon_start, epsilon_end, epsilon_decay):
        self.action = action
        self.reward = reward
        self.state = State(epsilon_start, epsilon_end, epsilon_decay)
        
    # Choose forward, left, right, or nothing using 
    def chooseAction(self, epsilon_start, epsilon_end):

    # Do Action w/ action variable
    def doAction(self, action):

    # Updates the stats
    def update(self, action, reward):
