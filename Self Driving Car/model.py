import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

## Approximate Q Function
class MLP(nn.Module):
    def __init__(self, input_size, num_of_actions):
        super().__init__()
        self.input_size = input_size
        self.num_of_actions = num_of_actions
        self.FC1 = nn.Linear(input_size, 50)
        self.FC2 = nn.Linear(50, num_of_actions)
    
    def forward(self, state):
        x = F.relu(self.FC1(state))
        q_value = self.FC2(x)
        return q_value


## Experience Replay (Memory)
class Replay(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, event): # event = (state, action, reward, next_state, done)
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0] # Remove the oldest memory to avoid large memory and computations
        
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size)) # Randomly takes a batch of experiences for training the agent
        return map(lambda x: Variable(torch.cat(x,0)), samples) # Convert all the samples into Pytorch 'Variable'
    

class DQN():

    def __init__(self, input_size, num_of_actions, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = MLP(input_size, num_of_actions)
        self.memory = Replay(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0) # state of agent from previous timestep
        self.last_action = 0
        self.last_reward = 0

    def select_action(self, state):
        '''
        The torch.nn.functional.softmax function in PyTorch converts a tensor of raw values into a probability distribution, where each element is between 0 and 1 and the elements sum to 1. torch.multinomial then samples from this distribution. It takes the probability distribution as input and returns indices, where the index i is sampled with probability proportional to the i-th element of the input tensor. 
        Here's how they are typically used together: 

        • Calculate Probabilities: The softmax function is applied to the output of a neural network layer (often the final layer in a classification model) to obtain probabilities for each class. 
        • Sample from Distribution: The torch.multinomial function is then used to sample from this probability distribution. This is often used in scenarios where you want to make a random choice based on the predicted probabilities, such as in reinforcement learning or generative models. 

        import torch
        import torch.nn.functional as F

        # Example: Softmax followed by multinomial sampling
        logits = torch.tensor([2.0, 1.0, 0.1])  # Example raw scores from a model
        probs = F.softmax(logits, dim=0)  # Convert to probabilities
        print("Probabilities:", probs)

        # Sample one index based on the probabilities
        num_samples = 1
        sampled_indices = torch.multinomial(probs, num_samples, replacement=True) 
        print("Sampled index:", sampled_indices)

        # Sample multiple indices
        num_samples = 5
        sampled_indices = torch.multinomial(probs, num_samples, replacement=True)
        print("Sampled indices:", sampled_indices)

        replacement=True means that the same index can be sampled multiple times. If replacement=False, each index can be sampled at most once, and num_samples must be less than or equal to the number of elements in probs. 

        '''
        probs = F.softmax(self.model(Variable(state, volatile=True))*100)
        action = probs.multinomial(1)
        return action.data[0,0]
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1,batch_action.unsqueeze(1)).squeeze(1) # q value
        next_outputs = self.model(batch_next_state).detach().max(1)[0] # max q value
        target = self.gamma*next_outputs + batch_reward # discounted term
        td_loss = F.smooth_l1_loss(outputs, target) # Temporal Difference loss (Huber Loss). Huber Loss is less sensitive to outliers than MSE
        self.optimizer.zero_grad()
        td_loss.backward()
        self.optimizer.step()