from os import stat
import torch
import torch.nn as nn
import numpy as np
from replay_buffer import *
from qnetwork import *

class DDQN:
    def __init__(self, buffer_size, state_size, q_hidden_sizes, action_size, batch_size,\
                 gamma, epsilon, tau, optimizer, update_step, criterion, lr, device): # TODO: su an duz epsilonlu falan yapicaz explorationi ama sonra guncellemek lazim iste bunu

        # Create the local and target networks and the replay buffer
        self.q_local = QNetwork(state_size, q_hidden_sizes, action_size)
        self.q_target = QNetwork(state_size, q_hidden_sizes, action_size)

        self.replay_buffer = Buffer(buffer_size)

        # Assign the class variables
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon # For expsilon greedy - will not be used later
        self.tau = tau
        self.optimizer = optimizer
        self.udpate_step = update_step # Number of steps to update target network 
        self.criterion = criterion # TODO: bunlari daha bi basitlestirsen mi kiz 
        self.lr = lr
        self.device = device
        
