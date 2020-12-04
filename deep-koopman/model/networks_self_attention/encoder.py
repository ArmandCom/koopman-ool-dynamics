import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
# from spectral import SpectralNorm
from .self_attention import Self_Attn

# class Discriminator(nn.Module):
#     """Discriminator, Auxiliary Classifier."""
#
#     def __init__(self, batch_size=64, image_size=64, conv_dim=64):
#         super(Discriminator, self).__init__()
#         self.imsize = image_size
#         layer1 = []
#         layer2 = []
#         layer3 = []
#         last = []
#
#         layer1.append(SpectralNorm(nn.Conv2d(3, conv_dim, 4, 2, 1)))
#         layer1.append(nn.LeakyReLU(0.1))
#
#         curr_dim = conv_dim
#
#         layer2.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
#         layer2.append(nn.LeakyReLU(0.1))
#         curr_dim = curr_dim * 2
#
#         layer3.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
#         layer3.append(nn.LeakyReLU(0.1))
#         curr_dim = curr_dim * 2
#
#         if self.imsize == 64:
#             layer4 = []
#             layer4.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
#             layer4.append(nn.LeakyReLU(0.1))
#             self.l4 = nn.Sequential(*layer4)
#             curr_dim = curr_dim * 2
#         self.l1 = nn.Sequential(*layer1)
#         self.l2 = nn.Sequential(*layer2)
#         self.l3 = nn.Sequential(*layer3)
#
#         last.append(nn.Conv2d(curr_dim, 1, 4))
#         self.last = nn.Sequential(*last)
#
#         self.attn1 = Self_Attn(256, 'relu')
#         self.attn2 = Self_Attn(512, 'relu')
#
#     def forward(self, x):
#         out = self.l1(x)
#         out = self.l2(out)
#         out = self.l3(out)
#         out, p1 = self.attn1(out)
#         out = self.l4(out)
#         out, p2 = self.attn2(out)
#         out = self.last(out)
#
#         return out.squeeze(), p1, p2

class ImageEncoder(nn.Module):
    #TODO: Add spatiotemporal attention
    def __init__(self, in_channels, feat_dim, n_objects, ngf, n_layers):
        super(ImageEncoder, self).__init__()

        activation = nn.ReLU(True)
        # activation = nn.ELU(True)

        self.n_objects = n_objects
        self.feat_dim = feat_dim
        ch = 32
        self.resolution = (32, 32)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, ch, 5, 1, 2),  # B,  32, 32, 32
            activation,
            nn.Conv2d(ch, ch, 5, 2, 2),  # B,  32, 16, 16
            activation,
            # Self_Attn(ch, 'relu'),
        )
        # self.cnn2 = nn.Sequential(
        #     nn.Conv2d(ch, ch, 5, 2, 2),  # B,  64,  8,  8
        #     activation,
        #     nn.Conv2d(ch, ch, 5, 2, 2),  # B,  64,  4,  4
        #     activation,
        #     nn.Conv2d(ch, ch, 5, 2, 2),  # B, 256,  1,  1
        #     activation,
        #     Print_Shape(),
        #     nn.Conv2d(ch, ch, 5, 2, 2),  # B, 256,  1,  1
        #     activation,
        #     nn.Conv2d(ch, n_objects * feat_dim, 5, 2, 2),  # B, 256,  1,  1
        #     activation
        # )
        # nn.ReplicationPad3d((1, 1, 2, 2, 2, 2))
        self.cnn3 = nn.Sequential(
            nn.Conv3d(ch, ch, [3, 5, 5], [1, 2, 2], [1, 2, 2]),
            # nn.Conv3d(ch, ch, [5, 9, 9], [1, 2, 2], [2, 4, 4]),
            activation,
            nn.Conv3d(ch, ch, [3, 5, 5], [1, 2, 2], [1, 2, 2]),
            activation,
            nn.Conv3d(ch, ch, [3, 5, 5], [1, 2, 2], [1, 2, 2]),
            activation,
            nn.Conv3d(ch, ch, [3, 5, 5], [1, 2, 2], [1, 2, 2]),
            activation,
            nn.Conv3d(ch, n_objects * feat_dim, [3, 5, 5], [1, 2, 2], [1, 2, 2]),
            activation
        )
        # x = x.permute(0, 2, 1, 3, 4)
        # x = self.cnn3(x)
        # x = x.permute(0, 2, 1, 3, 4)
        # x = x.reshape(-1, *x.shape[2:])

        self.linear_pos = nn.Linear(4, ch)
        self.register_buffer('grid_enc', self._build_grid(self.resolution))
        self.encoder_pos = self._soft_position_embed

        self.norm = nn.LayerNorm(ch)

        self.mlp = nn.Sequential(nn.Linear(ch, ch),
                                 activation,
                                 nn.Linear(ch, ch))

        # self.slot_attention = SlotAttention(n_objects, feat_dim)

    def _soft_position_embed(self, input):
        return input + self.linear_pos(self.grid_enc)

    def _build_grid(self, resolution):
        ranges = [np.linspace(0., 1., num=res) for res in resolution]
        grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
        grid = np.stack(grid, axis=-1)
        grid = np.reshape(grid, [resolution[0], resolution[1], -1])
        grid = np.expand_dims(grid, axis=0)
        grid = grid.astype(np.float32)
        return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1)).reshape(grid.shape[0], -1, 4)

    def forward(self, x, reverse=False):
        bs, T, ch, h, w = x.shape

        x = x.reshape(-1, *x.shape[2:])
        x = self.cnn(x)
        x = spatial_flatten_for_sa(x)
        x = self.encoder_pos(x)
        x = self.mlp(x) #self.norm(x)
        # TODO: check out if the norm is fucking something up

        # if reverse:
        #     x_rev = torch.flip(x.reshape(bs, T, *x.shape[1:]), dims=[1])
        #     x = x.reshape(bs, T, *x.shape[1:])
        #     x = torch.stack([x, x_rev], dim=1)
        #     bs = 2*bs
        #     x = x.reshape(bs * T, *x.shape[3:])

        x = x.reshape(bs, T, *x.shape[1:]).permute(0,3,1,2)
        x = x.reshape(*x.shape[:-1], int(np.sqrt(x.shape[-1])), int(np.sqrt(x.shape[-1])))
        x = self.cnn3(x)
        x = x.permute(0, 2, 1, 3, 4)\
            .reshape(bs, T, self.n_objects, self.feat_dim)\
            .permute(0, 2, 1, 3).reshape(bs * self.n_objects * T, self.feat_dim)

        return x

def spatial_flatten_for_sa(x):
  x = x.permute(0,2,3,1)
  return x.reshape(-1, x.shape[1] * x.shape[2], x.shape[-1])

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class Print_Shape(nn.Module):
    def __init__(self):
        super(Print_Shape, self).__init__()
        self.count = 0
    def forward(self, tensor):
        if self.count == 0:
            print('Tensor shape: ', tensor.shape)
            self.count += 1
        return tensor