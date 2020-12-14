import torch
from torch import nn


ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU
}


class ConvBlock(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, activation=nn.ReLU):
        super().__init__()
        self.add_module(
            "conv", nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False)
        )
        self.add_module(
            "bn", nn.BatchNorm2d(out_channels)
        )
        self.add_module(
            "act", activation()
        )


class ResidualBlock(nn.Module):

    def __init__(self, channels, kernel_size, stride=1, activation=nn.ReLU):
        super().__init__()
        self.conv_block_1 = ConvBlock(channels, channels, kernel_size, stride, activation)
        self.conv_block_2 = ConvBlock(channels, channels, kernel_size, stride, nn.Identity)
        self.act = activation()

    def forward(self, x):
        y = self.conv_block_1(x)
        y = self.conv_block_2(y)
        x = x + y
        x = self.act(x)
        return x


class DQN(nn.Module):

    def __init__(self, n_blocks=3, n_channels=64, activation="leaky_relu", tanh_nonlinearity=False):
        super().__init__()
        activation = ACTIVATIONS[activation]
        self.body = nn.Sequential(
            ConvBlock(3, n_channels, kernel_size=3, activation=activation),
            *[ResidualBlock(n_channels, kernel_size=3, activation=activation) for _ in range(n_blocks)]
        )
        self.head = nn.Sequential(
            nn.Conv2d(n_channels, 1, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, state_tensor):
        raw_Q = self.body(state_tensor)
        raw_Q = self.head(raw_Q).squeeze(1)
        board_tensor = state_tensor[:,0,...]
        empty_cells = (board_tensor == 0)
        Q = torch.ones_like(raw_Q)*(-float("inf"))
        Q[empty_cells] = raw_Q[empty_cells]
        return Q


class DuelingDQN(nn.Module):

    def __init__(self, n_blocks=3, n_channels=64, activation="leaky_relu"):
        super().__init__()
        activation = ACTIVATIONS[activation]
        self.body = nn.Sequential(
            ConvBlock(3, n_channels, kernel_size=3, activation=activation),
            *[ResidualBlock(n_channels, kernel_size=3, activation=activation) for _ in range(n_blocks)]
        )
        self.global_features = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            ConvBlock(n_channels, n_channels, kernel_size=1),
        )
        self.value_head = nn.Conv2d(n_channels, 1, kernel_size=1)
        self.advantage_head = nn.Conv2d(2*n_channels, 1, kernel_size=1)
        self.tanh = nn.Tanh()

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

        raw_Q = V.expand(A.shape) + A
        raw_Q = self.tanh(raw_Q)
        Q = torch.ones_like(raw_Q)*(-float("inf"))
        Q[empty_spaces] = raw_Q[empty_spaces]

        return Q


class AlphaZero(nn.Module):

    def __init__(self, n_blocks=3, n_channels=64, activation="leaky_relu"):
        super().__init__()
        activation = ACTIVATIONS[activation]
        self.body = nn.Sequential(
            ConvBlock(3, n_channels, kernel_size=3, activation=activation),
            *[ResidualBlock(n_channels, kernel_size=3, activation=activation) for _ in range(n_blocks)]
        )
        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1,-1),
            nn.Linear(n_channels, n_channels),
            nn.BatchNorm1d(n_channels),
            activation(),
            nn.Linear(n_channels, 1),
            nn.Tanh()
        )
        self.policy_head = nn.Sequential(
            ConvBlock(n_channels, n_channels, kernel_size=1, activation=activation),
            nn.Conv2d(n_channels, 1, kernel_size=1)
        )

    def forward(self, state_tensor):
        features = self.body(state_tensor)
        value_tensor = self.value_head(features).squeeze(1)
        raw_logits_tensor = self.policy_head(features).squeeze(1)

        logits_tensor = torch.ones_like(raw_logits_tensor)*(-float("inf"))
        board_tensor = state_tensor[:,0,...]
        empty_cells = (board_tensor == 0)

        logits_tensor[empty_cells] = raw_logits_tensor[empty_cells]

        return value_tensor, logits_tensor
