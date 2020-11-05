import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from model.networks.slot_attention import SlotAttention
import math

class AttImageEncoder(nn.Module):
    '''
    Encodes images. Similar structure as DCGAN.
    '''

    def __init__(self, in_channels, feat_dim, ngf, n_layers):
        super(AttImageEncoder, self).__init__()

        feat_size = [16, 16]
        n_inner_layers = 3

        # layers = [nn.Conv2d(in_channels, ngf, 7, 2, 3, bias=False),
        #           nn.ReLU(True)]  # nn.LeakyReLU(0.2, inplace=True)]
        # layers += [nn.Conv2d(ngf, ngf * 2, 5, 1, 2, bias=False),
        #           nn.BatchNorm2d(ngf * 2),
        #           nn.ReLU(True)]  # nn.LeakyReLU(0.2, inplace=True)]

        layers = [nn.Conv2d(in_channels, ngf, 5, 1, 2, bias=False),
                  nn.ReLU(True)]

        for i in range(0, n_inner_layers):
            layers += [nn.Conv2d(ngf, ngf * 2, 3, 1, 1, bias=False),
                       nn.BatchNorm2d(ngf * 2),
                       nn.ReLU(True)]  # nn.LeakyReLU(0.2, inplace=True)]
            ngf *= 2

        n_last_layers = int(math.log2(feat_size[0]))
        for i in range(0, n_layers - n_last_layers + 1):
            layers += [nn.Conv2d(ngf, ngf * 2, 5, 2, 2, bias=False),
                       nn.BatchNorm2d(ngf * 2),
                       nn.ReLU(True)]  # nn.LeakyReLU(0.2, inplace=True)]
            ngf *= 2

        layers += [nn.Conv2d(ngf, feat_dim, 3, 1, 1, bias=False)]

        self.main = nn.Sequential(*layers)

        self.norm = nn.LayerNorm(feat_size)

        self.mlp = nn.Sequential(nn.Conv2d(feat_dim + 4, feat_dim, 1, bias=False),
                                 nn.ReLU(True),
                                 nn.Conv2d(feat_dim, feat_dim * 2, 1, bias=False))

        # Reduce to flat
        layers_last = []
        for i in range(0, n_last_layers-1):
            layers_last += [nn.Conv2d(feat_dim * 2, feat_dim * 2, 4, 2, 1, bias=False),
                       nn.BatchNorm2d(feat_dim * 2),
                       nn.ReLU(True)]  # nn.LeakyReLU(0.2, inplace=True)]
        layers_last += [nn.Conv2d(feat_dim * 2, feat_dim * 2, 2, 2, 0, bias=False)]
        self.last_conv = nn.Sequential(*layers_last)

        self.last_pool = nn.MaxPool2d(kernel_size=feat_size)

        # Note: add coordinate
        # Positional encoding buffers
        x_l = torch.linspace(0, 1, feat_size[0])
        y_l = torch.linspace(0, 1, feat_size[1])
        x_r = x_l.flip(0)
        y_r = y_l.flip(0)
        x_grid_l, y_grid_l = torch.meshgrid(x_l, y_l)
        x_grid_r, y_grid_r = torch.meshgrid(x_r, y_r)
        # Add as constant, with extra dims for N and C
        self.register_buffer('x_grid_l_enc', x_grid_l.view((1, 1) + x_grid_l.shape))
        self.register_buffer('y_grid_l_enc', y_grid_l.view((1, 1) + y_grid_l.shape))
        self.register_buffer('x_grid_r_enc', x_grid_r.view((1, 1) + x_grid_r.shape))
        self.register_buffer('y_grid_r_enc', y_grid_r.view((1, 1) + y_grid_r.shape))

        # Note: Attention mechanism
        # self.slot_attention = SlotAttention(num_slots=2,
        #                                     dim=reduce((lambda x, y: x * y), feat_size),
        #                                     iters=3)

    def _add_positional_encoding(self, x):
        x = torch.cat((self.x_grid_l_enc.expand(x.shape[0], -1, -1, -1),
                       self.y_grid_l_enc.expand(x.shape[0], -1, -1, -1),
                       self.x_grid_r_enc.expand(x.shape[0], -1, -1, -1),
                       self.y_grid_r_enc.expand(x.shape[0], -1, -1, -1), x), dim=-3)
        return x

    def forward(self, x):

        f = self.main(x)
        fpe = self._add_positional_encoding(f)
        norm_fpe = self.norm(fpe)
        f = self.mlp(norm_fpe)

        # Attention mechanism
        # sa = self.slot_attention(f.reshape(*f.shape[:2], -1))

        # o = self.last_pool(f).squeeze(-1).squeeze(-1) #Option 1
        o = self.last_conv(f).squeeze(-1).squeeze(-1) #Option 2
        return o
