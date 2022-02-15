import torch.nn as nn
import torch.nn.functional as F
import torch
import random
import numpy as np

class Agent():
  
    def __init__(self, env, eps=1.0):
        self.alpha = 0.0025
        self.eps = eps
        self.eps_end = 0.01
        self.eps_incr = 0.9995
        self.gamma = 0.99
        self.env = env

        
        self.state_space_size = self.env.observation_space.shape[0] # 4
        self.action_space_size = self.env.action_space.n # 2
        self.model = QNet(input_size=4, alpha=self.alpha)
            
    def choose_action(self, state):
        
        rand = random.uniform(0, 1)
        if rand < self.eps:
            action = self.env.action_space.sample()
        else:
            state = torch.tensor(state, dtype = torch.float).to('cuda')
            q_sa_values, hidden = self.model(state)
            action = torch.argmax(q_sa_values).item()
            
        return action
    
    def train_(self, states_list, actions_list, rewards_list, states_next_list, dones_list):
        self.model.train()
        self.model.optimizer.zero_grad()
        
        q_list = []
        h = None

        for i in states_list:
            i = torch.tensor(i, dtype = torch.float).to('cuda')
            q, h = self.model(i,h)
            q_list.append(q.squeeze(0))
        q_list = torch.stack(q_list,1).squeeze(0)
        
        q_next_list = []
        h = None
        
        for i in states_next_list:
            i = torch.tensor(i, dtype = torch.float).to('cuda')
            q_next, h = self.model(i,h)
            q_next_list.append(q_next.squeeze(0))

        q_next_list = torch.stack(q_next_list,1).squeeze(0)
        
        actions = torch.tensor(actions_list).to('cuda')
        rewards = torch.tensor(rewards_list, dtype = torch.float).to('cuda')
        dones = torch.tensor(dones_list).to('cuda')
        indices = torch.arange(len(states_list)).to(self.model.device)

        q = q_list[indices, actions]
        q_next_list = q_next_list.max(dim=1)[0]
        q_next_list[dones] = 0.0
        q_target = rewards + 0.95*q_next_list

        loss = self.model.loss(q_target.float(),q.float()).to(self.model.device)#/len(states_list)
        loss.backward()
        self.model.optimizer.step()

        return loss.item()
    
    def change_eps(self):
        self.eps = max(self.eps_end, self.eps * self.eps_incr)
        
class QNet(nn.Module):
    def __init__(self, input_size,alpha):
        super().__init__()
        self.l1 = nn.Linear(input_size, 9)
        self.l2 = nn.Linear(9, 9)
        self.rnn = nn.RNN(input_size = 9,hidden_size = 3,num_layers = 1)
        
        self.l3 = nn.Linear(3, 2)
        
        self.optimizer = torch.optim.Adam(self.parameters(),lr=alpha)
        self.loss = nn.MSELoss()
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, x, h = None):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = x.reshape(1,1,9)

        x,h = self.rnn(x,h)

        x = self.l3(x)
        return x,h