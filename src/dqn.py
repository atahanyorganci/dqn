import torch
from torch import nn


def conv2d_size_out(size, kernel_size, stride):
    return (size - (kernel_size - 1) - 1) // stride + 1


class DQN(nn.Module):

    def __init__(self, state_shape, action_count):
        super(DQN, self).__init__()
        self.state_shape = state_shape
        self.action_count = action_count
        self.net = DQN.construct_net(state_shape[0], action_count)

    @classmethod
    def construct_net(cls, channels, actions):
        layers = [
            nn.Conv2d(in_channels=channels, out_channels=32,
                      kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, actions),
        ]
        return nn.Sequential(*layers)

    def forward(self, X):
        return self.net(X)

    def act(self, X):
        return torch.argmax(self.net(X), axis=1)
