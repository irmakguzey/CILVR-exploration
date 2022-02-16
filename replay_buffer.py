import collections
from dataclasses import replace
import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Buffer:
    def __init__(self, buffer_size):
        self.buffer = collections.deque(maxlen=buffer_size)
        self.buffer_size = buffer_size

    def add(self, state, action, reward, next_state, done):
        # NOTE: We add to the buffer everytime we have a new observation - so state is 1 element only
        self.buffer.append([state, action, reward, next_state, done])

        # print('----\nself.buffer: {}'.format(self.buffer))

    def sample(self, batch_size):
        # Get the random indices
        if batch_size > len(self.buffer):
            indices = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=True)
        else:
            indices = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)

        # print('indices: {}'.format(indices))

        # Put all together into a numpy array
        experiences = np.array(self.buffer)[indices]

        # print('experiences: {}'.format(experiences))

        # Get states and actions and etc. separately
        states = torch.from_numpy(np.stack(experiences[:,0], axis = 0)  ).to(device)
        actions = torch.from_numpy(np.stack(experiences[:,1], axis = 0)).to(device)
        rewards = torch.from_numpy(np.stack(experiences[:,2], axis = 0)).to(device)
        new_states = torch.from_numpy(np.stack(experiences[:,3], axis = 0)).to(device)
        dones = torch.from_numpy(np.stack(experiences[:,4], axis = 0)).to(device)

        # print('states: {}, actions: {}, rewards: {}, new_states: {}, dones: {}'.format(
        #     states, actions, rewards, new_states, dones
        # ))

        return states, actions, rewards, new_states, dones

    def __len__(self):
        return len(self.buffer)


# buffer = Buffer(10)

# for i in range(10):
#     rand_exp = np.random.rand(5)
#     print('rand_exp: {}'.format(rand_exp))
#     buffer.add(rand_exp[0], rand_exp[1], rand_exp[2], rand_exp[3], rand_exp[4])
#     buffer.sample(2)