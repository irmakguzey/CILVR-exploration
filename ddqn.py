from os import stat
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from replay_buffer import *
from qnetwork import *

class DDQN:
    def __init__(self, buffer_size, state_size, q_hidden_sizes, action_size, batch_size,\
                 gamma, epsilon, tau, update_step, criterion, lr, device): # TODO: su an duz epsilonlu falan yapicaz explorationi ama sonra guncellemek lazim iste bunu

        self.step_num = 0

        # Create the local and target networks and the replay buffer
        self.q_local = QNetwork(state_size, q_hidden_sizes, action_size)
        self.q_target = QNetwork(state_size, q_hidden_sizes, action_size)

        self.replay_buffer = Buffer(buffer_size)

        # Assign the class variables
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon # For expsilon greedy - will not be used later
        self.tau = tau
        self.update_step = update_step # Number of steps to update target network 
        self.criterion = criterion # TODO: bunlari daha bi basitlestirsen mi kiz 
        self.optimizer = optim.Adam(self.q_local.parameters(), lr=lr)
        self.device = device

    def step(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

        # Check if we should soft update the target network
        self.step_num += 1
        if self.step_num % self.update_step == 0:
            self.soft_update()

        if len(self.replay_buffer) > self.batch_size: # Get experiments
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
            loss = self.learn(states, actions, rewards, next_states, dones)
            return loss

    def act(self, state):
        self.q_local.eval()
        with torch.no_grad():
            state = torch.from_numpy(state).to(device)
            all_actions = self.q_local(state) # Get the q values for all actions
            # print('all_actions: {}'.format(all_actions))
        
        self.q_local.train() # Set the network mode to training
        
        if np.random.uniform(0,1) < self.epsilon: # TODO: This will be different - i guess
            return np.random.choice(np.arange(self.action_size))
        else:
            return torch.argmax(all_actions, dim=1).item()

    def learn(self, states, actions, rewards, next_states, dones):
        # Method to train the q networks 
        self.q_local.train()
        self.q_target.eval() # Target network is used to get the error of the local network 

        preds = self.q_local(states).gather(1, actions.reshape(-1,1)) # Bu actionlarin yapilmis hallerini aliyor - hangi beklentiyle o actioni yaptik gibi 

        with torch.no_grad():
            next_preds = self.q_target(next_states).detach().max(1)[0] # Bu da maxlari aldigi icin aslinda sonrasinda ne yapacagimizin tahmini aliyor gibi
            # TODO: count based'i burada eklemisler
        target = rewards + ((1-dones) * self.gamma * next_preds)

        # Calculate the loss 
        loss = self.criterion(preds.float(), target.reshape(-1,1).float()).to(self.device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def soft_update(self):
        # Update the target network
        for target_param, local_param in zip(self.q_local.parameters(), self.q_target.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1-self.tau) * target_param.data) # TODO: make sure this works - the deleted this part in the github code 
        # self.q_target.load_state_dict(self.q_local.state_dict())

        
#   def save(self) :
#     self.qnetwork_local.save_checkpoint()
#     self.qnetwork_target.save_checkpoint()
#     if self.hash is not None :
#       save_obj(self.hash.hash, os.path.join('models' ,self.name + '_hash'))
#   def load(self, hash = True) :
#     self.qnetwork_local.load_checkpoint()
#     self.qnetwork_target.load_checkpoint()
#     if self.hash is not None and hash :
#       self.hash.hash = load_obj(os.path.join(self.name + '_hash'))
