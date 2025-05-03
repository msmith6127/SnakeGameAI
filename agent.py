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
        self.n_games = 0
        self.epsilon = 0  # randomness for exploration
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = LinearQNet(11, 128, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        '''
        Gets the current state from the game environment.
        '''
        return game.get_state()

    def remember(self, state, action, reward, next_state, done):
        '''
        Stores a piece of memory to train on later; adds experience. 
        '''
        self.memory.append((state, action, reward, next_state, done))  # popleft() if full

    def train_long_memory(self):
        '''
        Train the model on past experiences; if it has too many to choose from,
        it should pick a random one. 
        '''
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        '''
        Trains the model on the most recent state for quick learning. 
        '''
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        '''
        This decides what the agent's move should make in its current state.
        It could either make a random move or decide on its own, whatever it thinks is best. 
        '''
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move, move
