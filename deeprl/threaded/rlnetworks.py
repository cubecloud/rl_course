import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

__version__ = 0.008


class FCnet(nn.Module):
    def __init__(self, state_size: int = 8, n_actions: int = 6, l1_filters: int = 512, l2_filters: int = 512,
                 seed: int = 42, ):
        """
        Torch fully connected network

        Args:
            state_size (int):   state (observation) size
            n_actions (int):    actions (quantity) size
            seed (int):         random seed
            l1_filters (int):   Layer 1 filters (units)
            l2_filters (int):   Layer 2 filters (units)
        """
        super(FCnet, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.layer1 = nn.Linear(state_size, l1_filters)
        self.layer2 = nn.Linear(l1_filters, l2_filters)
        self.layer3 = nn.Linear(l2_filters, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class EMBEDnet(nn.Module):
    def __init__(self,
                 vocab_size: int = 500,
                 embed_out: int = 4,
                 context_size: int = 1,
                 n_actions: int = 6,
                 l1_filters: int = 50,
                 seed: int = 42):
        """
        Torch Embedding network

        Args:
            embed_in (int):     state (observation) size
            embed_out (int):    internal embedding size
            n_actions (int):    actions (quantity) size
            seed (int):         random seed
            l1_filters (int):   Layer 1 filters (units)
            l2_filters (int):   Layer 2 filters (units)
        """
        self.context_size = context_size
        self.embed_out = embed_out
        super(EMBEDnet, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.emb = nn.Embedding(vocab_size, embed_out)
        self.layer1 = nn.Linear(embed_out * context_size, l1_filters)
        self.layer2 = nn.Linear(l1_filters, n_actions)

    def forward(self, x):
        embeds = self.emb(x).view((-1, self.embed_out * self.context_size))
        x = F.relu(self.layer1(embeds))
        return self.layer2(x)


class Conv2Dnet(nn.Module):
    def __init__(self, state_shape, n_actions, start_filters=32, fc_filters=512, seed=42):
        """
        Torch Conv2D network

        Args:
            state_shape (tuple):    state (observation) shape - 2D
            n_actions (int):        actions (quantity) size
            start_filters (int):    starting layer filters (units). The whole number is divisible by 8
                                    Default: 32
            seed (int):             random seed

        """
        super(Conv2Dnet, self).__init__()

        self.seed = torch.manual_seed(seed)
        self.conv_block = nn.Sequential(nn.Conv2d(state_shape[0], start_filters, kernel_size=8, stride=4),
                                        nn.ReLU(),
                                        nn.Conv2d(start_filters, start_filters * 2, kernel_size=4, stride=2),
                                        nn.ReLU(),
                                        nn.Conv2d(start_filters * 2, start_filters * 2, kernel_size=3, stride=1),
                                        nn.ReLU()
                                        )

        conv_out_size = self._get_conv_out(state_shape)
        self.fc_block = nn.Sequential(nn.Linear(conv_out_size, fc_filters),
                                      nn.ReLU(),
                                      nn.Linear(fc_filters, n_actions)
                                      )

    def _get_conv_out(self, shape):
        o = self.conv_block(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv_block(x).view(x.size()[0], -1)
        return self.fc_block(conv_out)


class ActorNet(nn.Module):
    """Actor Network for Policy Gradient Methods."""

    def __init__(self, state_size: int = 4, l1_filters: int = 16, l2_filters: int = 16, out_filters: int = 2,
                 seed: int = 42, last_activation=None):
        """
        Initializes the ActorNet class with customizable parameters.

        Args:
            state_size (int): Input size of the network (default: 4).
            l1_filters (int): Number of filters in the first layer (default: 16).
            l2_filters (int): Number of filters in the second layer (default: 16).
            out_filters (int): Output size of the network (default: 2).
            seed (int): Random seed for reproducibility (default: 42).
            last_activation (Callable, optional): Activation function to apply at the output layer.
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)  # Set random seed for reproducibility
        self.layer1 = nn.Linear(state_size, l1_filters)  # First linear layer
        self.layer2 = nn.Linear(l1_filters, l2_filters)  # First linear layer

        # Create the second layer based on whether an activation function was provided
        if last_activation == 'softmax':
            self.layer3 = nn.Sequential(
                nn.Linear(l2_filters, out_filters),  # Third linear layer
                nn.Softmax(dim=1)  # Apply Softmax along the classes axis
            )
        elif last_activation is not None:
            self.layer3 = nn.Sequential(
                nn.Linear(l2_filters, out_filters),  # Third linear layer
                last_activation()  # Apply specified activation function
            )
        else:
            self.layer3 = nn.Sequential(
                nn.Linear(l2_filters, out_filters)  # Third linear layer without activation
            )

    def forward(self, x: torch.Tensor):
        """
        Performs a forward pass through the network.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying layers and activations.
        """
        x = F.relu(self.layer1(x))  # Apply ReLU activation to the first layer's output
        x = F.relu(self.layer2(x))  # Apply ReLU activation to the second layer's output
        return self.layer3(x)  # Return the result from the second layer


class ContinuousActorNet(nn.Module):
    def __init__(self, input_shape: Tuple[int, int, int] = (4, 96, 96), l1_filters: int = 8, action_dim: int = 3,
                 features_dim: int = 100, seed: int = 42):
        super(ContinuousActorNet, self).__init__()
        self.seed = torch.manual_seed(seed)  # Set random seed for reproducibility
        self.cnn_base = nn.Sequential(  # input shape (4, 96, 96)
            nn.Conv2d(input_shape[0], l1_filters, kernel_size=4, stride=2),
            nn.ReLU(),  # activation
            nn.Conv2d(l1_filters, l1_filters * 2, kernel_size=3, stride=2),  # (8, 47, 47)
            nn.ReLU(),  # activation
            nn.Conv2d(l1_filters * 2, l1_filters * 4, kernel_size=3, stride=2),  # (16, 23, 23)
            nn.ReLU(),  # activation
            nn.Conv2d(l1_filters * 4, l1_filters * 8, kernel_size=3, stride=2),  # (32, 11, 11)
            nn.ReLU(),  # activation
            nn.Conv2d(l1_filters * 8, l1_filters * 16, kernel_size=3, stride=1),  # (64, 5, 5)
            nn.ReLU(),  # activation
            nn.Conv2d(l1_filters * 16, l1_filters * 32, kernel_size=3, stride=1),  # (128, 3, 3)
            nn.ReLU(),  # activation
        )  # output shape (256, 1, 1)

        # Calculate combined feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, input_shape[0], input_shape[1], input_shape[2])
            d = self.cnn_base(dummy_input)
            d = d.view(d.size(0), -1)
            self.conv2d_features = d.shape[-1]
            # print(f'Features from conv2d extractor: {self.conv2d_features}')
        self.fc = nn.Sequential(nn.Linear(self.conv2d_features, features_dim), nn.ReLU())

        with torch.no_grad():
            d = self.fc(d)
            self._features_dim = d.shape[-1]
            # print(f'features_dim: {d.shape[-1]}')

        self.v = nn.Sequential(nn.Linear(self._features_dim, self._features_dim),
                               nn.ReLU(),
                               nn.Linear(self._features_dim, 1))
        self.mu_layer = nn.Sequential(nn.Linear(self._features_dim, action_dim),
                                      nn.Tanh())  # [-1., 1.] range
        self.sigma_layer = nn.Sequential(nn.Linear(self._features_dim, action_dim),
                                         nn.Softplus())
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        x = self.cnn_base(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        v = self.v(x)
        mu = self.mu_layer(x)
        sigma = self.sigma_layer(x) + 1e-5  # positive sigma
        return (mu, sigma), v
