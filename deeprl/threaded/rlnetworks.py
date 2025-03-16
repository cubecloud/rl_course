import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def __init__(self, state_size: int = 4, l1_filters: int = 16, out_filters: int = 2,
                 seed: int = 42, last_activation=None):
        """
        Initializes the ActorNet class with customizable parameters.

        Args:
            state_size (int): Input size of the network (default: 4).
            l1_filters (int): Number of filters in the first layer (default: 16).
            out_filters (int): Output size of the network (default: 2).
            seed (int): Random seed for reproducibility (default: 42).
            last_activation (Callable, optional): Activation function to apply at the output layer.
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)  # Set random seed for reproducibility
        self.layer1 = nn.Linear(state_size, l1_filters)  # First linear layer

        # Create the second layer based on whether an activation function was provided
        if last_activation == 'softmax':
            self.layer2 = nn.Sequential(
                nn.Linear(l1_filters, out_filters),  # Third linear layer
                nn.Softmax(dim=1)  # Apply Softmax along the classes axis
            )
        elif last_activation is not None:
            self.layer2 = nn.Sequential(
                nn.Linear(l1_filters, out_filters),  # Third linear layer
                last_activation()  # Apply specified activation function
            )
        else:
            self.layer2 = nn.Sequential(
                nn.Linear(l1_filters, out_filters)  # Third linear layer without activation
            )

    def forward(self, x):
        """
        Performs a forward pass through the network.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying layers and activations.
        """
        x = F.relu(self.layer1(x))  # Apply ReLU activation to the first layer's output
        return self.layer2(x)  # Return the result from the second layer
