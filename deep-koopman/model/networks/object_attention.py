import numpy as np

import torch
from torch import nn

class ObjectAttention(nn.Module):
    """CNN encoder, maps observation to obj-specific feature maps."""

    def __init__(self, input_dim, hidden_dim, num_objects, input_size=[64, 64], act_fn='sigmoid',
                 act_fn_hid='relu'):
        super(ObjectAttention, self).__init__()

        self.cnn1 = nn.Conv3d(input_dim, hidden_dim, [3, 5, 5], padding=[1, 2, 2])
        self.act1 = get_act_fn(act_fn_hid)
        self.ln1 = nn.BatchNorm3d(hidden_dim)

        self.cnn2 = nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1)
        self.act2 = get_act_fn(act_fn_hid)
        self.ln2 = nn.BatchNorm3d(hidden_dim)

        self.cnn3 = nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1)
        self.act3 = get_act_fn(act_fn_hid)
        self.ln3 = nn.BatchNorm3d(hidden_dim)

        self.cnn4 = nn.Conv3d(hidden_dim, num_objects, 3, padding=1)
        self.act4 = get_act_fn(act_fn)

    def forward(self, obs):
        h = self.act1(self.ln1(self.cnn1(obs)))
        h = self.act2(self.ln2(self.cnn2(h)))
        # h = self.act3(self.ln3(self.cnn3(h)))

        # h = self.act1(self.cnn1(obs))
        # h = self.act2(self.cnn2(h))
        # h = self.act3(self.cnn3(h))
        return self.act4(self.cnn4(h))
