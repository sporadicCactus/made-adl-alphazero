import torch
from torch import nn


ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU
}


def ConvBlock(in_channels, out_channels, kernel_size, stride=1, activation=nn.ReLU):
    block = nn.Sequential()
    block.add_module(
        "conv", nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False)
    )
    block.add_module(
        "bn", nn.BatchNorm2d(out_channels)
    )
    block.add_module(
        "act", activation()
    )
    return block


class DQN(nn.Module):

    def __init__(self, hidden_sizes=[32,64,32], activation="leaky_relu"):
        super().__init__()
        activation = ACTIVATIONS[activation]
        sizes = [3] + hidden_sizes
        self.body = nn.Sequential(
            *[ConvBlock(in_c, out_c, kernel_size=3, activation=activation) for in_c, out_c in zip(sizes[:-1], sizes[1:])],
        )
        self.head = nn.Conv2d(sizes[-1], 1, kernel_size=1)

    def forward(self, state_tensor):
        Q = self.body(state_tensor)
        Q = self.head(Q).squeeze(1)
        board_tensor = state_tensor[:,0,...]
        Q[board_tensor != 0] = -float("inf")
        return Q


class DuelingDQN(nn.Module):

    def __init__(self, hidden_sizes=[32,64,32], activation="leaky_relu"):
        super().__init__()
        activation = ACTIVATIONS[activation]
        sizes = [3] + hidden_sizes
        self.body = nn.Sequential(
            *[ConvBlock(in_c, out_c, kernel_size=3, activation=activation) for in_c, out_c in zip(sizes[:-1], sizes[1:])],
        )
        self.global_features = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            ConvBlock(sizes[-1], sizes[-1], kernel_size=1),
        )
        self.value_head = nn.Conv2d(sizes[-1], 1, kernel_size=1)
        self.advantage_head = nn.Conv2d(2*sizes[-1], 1, kernel_size=1)

    def forward(self, state_tensor):
        features = self.body(state_tensor)
        global_features = self.global_features(features)
        V = self.value_head(global_features).squeeze(1)
        A = self.advantage_head(
                torch.cat([features, global_features.expand(features.shape)], dim=1)
        ).squeeze(1)

        board_tensor = state_tensor[:,0,...]
        empty_spaces = (board_tensor == 0)
        n_empty_spaces = empty_spaces.flatten(1,-1).sum(dim=-1)

        A[torch.logical_not(empty_spaces)] = 0
        average_advantage = A.flatten(1,-1).sum(dim=-1)/torch.max(n_empty_spaces, torch.ones_like(n_empty_spaces))
        V = V - average_advantage[:,None,None]

        Q = V.expand(A.shape) + A
        Q[torch.logical_not(empty_spaces)] = -float("inf")

        return Q
