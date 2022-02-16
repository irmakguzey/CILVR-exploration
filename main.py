# Script to merge every class and train the agent with different configurations 

from math import gamma
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ddqn import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPISODE_NUM = 100
TRAIN_EVERY = 10 # TODO: check this 
SAVE_EVERY = 1 # Save the loss in every 50 episodes
BATCH_SIZE = 64
BUFFER_SIZE = 100000
STATE_SIZE = 2 # Observation space is 2 dimensional in MountainCar environment
ACTION_SIZE = 3 # Accelerate left or right or stop and do nothing
GAMMA = 0.9 # TODO: check this 
EPSILON = 0.3
TAU = 0.98
UPDATE_STEP = 10
LEARNING_RATE = 0.01 # TODO: check this



criterion = nn.MSELoss()
hidden_sizes = [64, 32]
ddqn_agent = DDQN(
    buffer_size = BUFFER_SIZE, state_size=STATE_SIZE, q_hidden_sizes=hidden_sizes,
    action_size=ACTION_SIZE, batch_size=BATCH_SIZE, gamma=GAMMA, epsilon=EPSILON, tau=TAU,
    update_step=UPDATE_STEP, criterion=criterion, lr=LEARNING_RATE, device=device
)

losses = []
env = gym.make('MountainCar-v0') # Continuous mountain car cannot be trained with q learning since it doesn't have discrete action space 
state = env.reset()

for ep_num in range(EPISODE_NUM):
    action = ddqn_agent.act(state)
    next_state, reward, done, _ = env.step(action)
    loss = ddqn_agent.step(state, action, reward, next_state, 1.0*done)
    if loss:
        losses.append(loss)
    state = next_state

print('losses: {}'.format(losses))
# print('env.reset(): {}'.format(env.reset()))
# for ep_num in range(EPISODE_NUM):
    

# for _ in range(1000):
#     # env.render()
#     list = env.step(env.action_space.sample()) # take a random action
#     print('list: {}'.format(list))
# env.close()