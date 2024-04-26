import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__version__ = 0.003


class FCnet(nn.Module):
    def __init__(self, state_size: int, n_actions: int, seed: int = 42, l1_filters: int = 512, l2_filters: int = 512):
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
        # self.seed = torch.manual_seed(seed)
        self.layer1 = nn.Linear(state_size, l1_filters)
        self.layer2 = nn.Linear(l1_filters, l2_filters)
        self.layer3 = nn.Linear(l2_filters, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


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
                                        nn.Conv2d(start_filters, start_filters*2, kernel_size=4, stride=2),
                                        nn.ReLU(),
                                        nn.Conv2d(start_filters*2, start_filters*2, kernel_size=3, stride=1),
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
