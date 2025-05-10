import random
import numpy as np
from collections import deque
from model import LinearQNet, QTrainer
import torch

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    '''
    The Agent class defines the game agent which is based on Deep Q Learning and 
    will handle decision making and training.
    '''
    
    def __init__(self):
        """
        Initializes the agent with:
        - epsilon for exploration vs exploitation
        - gamma as the discount rate for future rewards
        - replay memory for storing experience tuples
        - a linear Q-network model and trainer
        - tracking metrics (loss, exploration, survival time, action types)
        """
        self.n_games = 0
        self.epsilon = 0  # randomness for exploration
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = LinearQNet(11, 128, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.survival_times = [] #tracks frames per game
        self.losses = [] # tracks training loss
        self.random_actions = 0 #count of random actions taken
        self.model_actions = 0 #count of model-based actions taken
        self.exploration_rates = [] #track epsilon over time
        

    def get_state(self, game):
         """
        Extracts the current state of the game environment.
        Args: game: The game environment instance
        Returns:A numpy array representing the state vector
        """
        return game.get_state()

    def remember(self, state, action, reward, next_state, done):
        '''
        Stores a piece of memory to train on later; adds experience. 
        Args:
            state: The current state
            action: Action taken
            reward: Reward received
            next_state: Resulting next state
            done: Boolean indicating if the episode ended
        '''
        self.memory.append((state, action, reward, next_state, done))  # popleft() if full

    def train_long_memory(self):
        '''
        Train the model on past experiences; if it has too many to choose from,
        it should pick a random one. 
        Returns: The training loss value.
        '''
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        loss = self.trainer.train_step(states, actions, rewards, next_states, dones)
        self.losses.append(loss)
        return loss

    def train_short_memory(self, state, action, reward, next_state, done):
        '''
        Trains the model on the most recent state for quick learning. 
        Args:
            state: Current state
            action: Action taken
            reward: Immediate reward
            next_state: Next observed state
            done: Boolean indicating episode end

        Returns: The training loss value.
        '''
        loss = self.trainer.train_step(state, action, reward, next_state, done)
        return loss

    def get_action(self, state):
        '''
        This decides what the agent's move should make in its current state.
        It could either make a random move or decide on its own, whatever it thinks is best. 
        Selects an action using an epsilon-greedy policy:
        - With probability epsilon, selects a random action (exploration)
        - Otherwise, chooses the best predicted action (exploitation)

        Args: state: Current game state as input features
        Returns:
            A tuple containing:
                - final_move: One-hot encoded list of the chosen action
                - move: Integer index of the action taken
        '''
        self.epsilon = 80 - self.n_games # decay epsilon
        self.epsilon = max(0, self.epsilon) # so epsilon doesn't go below 0
        
        # store the current exploration rate
        exploration_rate = self.epsilon/200
        
        final_move = [0, 0, 0] 

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
            self.random_actions = self.random_actions + 1 #increment random action counter
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
            self.model_actions = self.model_actions + 1 # increment model action counter

        return final_move, move
