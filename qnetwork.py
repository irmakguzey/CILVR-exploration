import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, state_size, hidden_sizes, action_size):
        super(QNetwork, self).__init__()

        self.state_size = state_size
        layer_list = [
            nn.Linear(state_size, hidden_sizes[0]),
            nn.ReLU()
        ]
        i = 0
        while i < len(hidden_sizes)-1:
            layer_list += [
                nn.Linear(hidden_sizes[i], hidden_sizes[i+1]),
                nn.ReLU()
            ]
            i += 1
        layer_list += [
            nn.Linear(hidden_sizes[i], action_size),
            nn.ReLU() # TODO: Make sure that this is what you really want to do
        ]

        # print('layer_list: {}'.format(layer_list))
        self.layers = nn.Sequential(*layer_list)
        # print('moduled layer_list: {}'.format(self.layers))

    def forward(self, x):
        # print('x before reshape: {}'.format(x))
        x = x.reshape(-1, self.state_size).float()
        # print('x after reshape: {}'.format(x))
        return self.layers(x)

q_network = QNetwork(5, [20, 32], 6)