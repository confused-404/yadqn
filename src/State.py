'''
This program is supposd to handle all of the data and turn it into the next action
'''

class State:
    def __init__(self, epsilon_start, epsilon_end, epsilon_decay):
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
    
    # Determines next action
    def forward(self):

    # Updates values
    def update(self, epsilon_start, epsilon_end, epsilon_decay):