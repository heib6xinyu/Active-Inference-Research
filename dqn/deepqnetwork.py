# %%
import torch
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gym
import math
import random
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import itertools
from collections import deque
# %%
env = gym.make('CartPole-v1')
# %%
env.reset()
state, reward, done, truncated, info = env.step(1)
print(state)
# %%
print(reward)
#1.0 mean not died
# %%
print(done)
# %%
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4,64),
            nn.Tanh(),
            nn.Linear(64,2)
        )
    def forward(self,t):
        t=self.net(t)
        return t
    def act(self, state):
        state_t = torch.as_tensor(state,dtype = torch.float32) 
        #we should turn it to tensor to pass through torch network
        q_values = self(state_t.unsqueeze(0))
# %%
def decay(eps_start, eps_end, eps_decayrate, current_step):
    return eps_end + (eps_start - eps_end)* np.exp(-1*eps_decayrate*current_step)
# %%
t = 5
eps = decay(1,0.001, 0.001,t)
eps
# %%
t = 500
eps = decay(1,0.001, 0.001,t)
eps

# %%
eps_count = 0 #it is like a counter of step
batch_size=32
gamma=0.99 #for bemman equation

online_net = Network() #for the first guess
target_net = Network() #the more educated guess
target_net.load_state_dict(online_net.state_dict())

epsilon_start = 1
epsilon_end = 0.001
epsilon_decayrate = 0.003

epsiode_durations = []

optimizer = torch.optim.Adam(online_net.parameters(), lr = 5e-4)

# %%
replayMemory = deque(maxlen=50000)#replay memory
state = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    s1,reward,done,truncated,_ = env.step(action)#state is the state 0
    experience = (state,action, reward, done,s1)
    replayMemory.append(experience)
    state = s1
    
    if done:
        env.reset()

for t in range(1000):# for 1000 epsiode
    state = env.state()

    for step in itertools.count(): #loop until epsiode end
        eps_count += 1
        epsilon = decay(epsilon_start,epsilon_end,epsilon_decayrate,eps_count)
        if random.random() <= epsilon:
            action = env.action_space.sample()
        else:
            action = online_net.act(state)
# %%
