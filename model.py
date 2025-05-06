import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LinearQNet(nn.Module):
    """
    This class defines the neural network model being used for the Q-function
    In other words, it's used to estimate the expected reward for each action
    """
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialization using:
        - input_size: number of features
        - hidden_size: number of neurons in the hidden layer
        - output_size: number of actions
        """
        super(LinearQNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        
        # weights for improved learning
        nn.init.kaiming_uniform_(self.linear1.weight)
        nn.init.kaiming_uniform_(self.linear2.weight)
        nn.init.kaiming_uniform_(self.linear3.weight)

    def forward(self, x):
        """
        Forward pass through the network:
        - x: represents the input state
        """
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def save(self, file_name='model.pth'):
        """
        Saves the weights of the model for later use
        """
        torch.save(self.state_dict(), file_name)


class QTrainer:
    """
    Trains the model using the Q-learning algorithm
    """
    def __init__(self, model, lr, gamma):
        """
        Initialization using:
        - model: the neural network model
        - lr: learning rate
        - gamma: discount factor (how much importance to give to future rewards)
        """
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        """
        Does one step of training using the Q-learning algorithm
        - state: current state(s)
        - action: action(s) taken
        - reward: reward(s) received
        - next_state: next state(s) after taking action(s)
        - done: indicates if the game is over (true) or not (false)
        """
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            # reshape for a single sample
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # Predict Q values for current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][action[idx]] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
