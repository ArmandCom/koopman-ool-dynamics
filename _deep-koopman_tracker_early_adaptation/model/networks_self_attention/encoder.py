import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
# from spectral import SpectralNorm
from .self_attention import Self_Attn, Temporal_Self_Attn
from .slot_attention import TemporalSlotAttention
from functools import reduce

class ImageEncoder(nn.Module):
    #TODO: Add spatiotemporal attention
    def __init__(self, in_channels, feat_dim, n_objects, ngf, n_layers):
        super(ImageEncoder, self).__init__()

        activation = nn.ReLU(True)
        # activation = nn.ELU(True)

        self.n_objects = n_objects
        self.feat_dim = feat_dim
        ch = 32
        self.resolution = (16, 16)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, ch, 5, 1, 2),  # B,  32, 64, 64
            activation,
            nn.Conv2d(ch, ch, 5, 2, 2),  # B,  32, 32, 32
            activation,
            nn.Conv2d(ch, ch, 5, 1, 2),  # B,  32, 32, 32
            activation,
            nn.Conv2d(ch, ch, 5, 2, 2),  # B,  32, 16, 16
            activation,
            nn.Conv2d(ch, ch, 5, 1, 2),  # B,  32, 16, 16
            activation,
        )

        att_dim = reduce((lambda x, y: x * y), self.resolution)

        self.cnn_ini = nn.Sequential(
            # nn.Conv2d(ch, ch, 5, 2, 2),  # B,  32, 16, 16
            # activation,
            nn.Conv2d(ch, ch, 5, 2, 2),  # B,  32, 8, 8
            activation,
            nn.Conv2d(ch, ch, 5, 2, 2),  # B,  32, 4, 4
            activation,
            nn.Conv2d(ch, ch, 5, 2, 2),  # B,  32, 2, 2
            activation,
            nn.Conv2d(ch, feat_dim * self.n_objects, 5, 2, 2),  # B,  32, 1, 1
            activation,
        )

        self.linear_pos = nn.Linear(4, ch)
        self.register_buffer('grid_enc', self._build_grid(self.resolution))
        self.encoder_pos = self._soft_position_embed

        self.norm = nn.LayerNorm(ch)
        # self.norm_2 = nn.LayerNorm(att_dim)

        self.mlp = nn.Sequential(nn.Linear(ch, ch),
                                 activation,
                                 nn.Linear(ch, ch))

        # self.mlp_2 = nn.Sequential(nn.Linear(att_dim, ch),
        #                          activation,
        #                          nn.Linear(ch, feat_dim))

        self.slot_attention = TemporalSlotAttention(self.n_objects, att_dim, feat_dim)

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
        # x = self.norm(x)
        x = self.mlp(x) #self.norm(x)

        # 3D Conv with or without attention
        x = x.permute(0, 2, 1)
        x = x.reshape(*x.shape[:-1], int(np.sqrt(x.shape[-1])), int(np.sqrt(x.shape[-1])))
        x = x.reshape(bs, T, *x.shape[1:])

        slots_ini = self.cnn_ini(x[:, 0])[..., 0, 0].reshape(bs, self.n_objects, -1)
        x = self.slot_attention(x, slots_ini)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(bs * self.n_objects * T, -1)
        # x = self.norm_2(x)
        # x = self.mlp_2(x)

        return x

# class ImageEncoder(nn.Module):
#     #TODO: Add spatiotemporal attention
#     def __init__(self, in_channels, feat_dim, n_objects, ngf, n_layers):
#         super(ImageEncoder, self).__init__()
#
#         activation = nn.ReLU(True)
#         # activation = nn.ELU(True)
#
#         self.n_objects = n_objects
#         self.feat_dim = feat_dim
#         ch = 32
#         self.resolution = (32, 32)
#         self.cnn = nn.Sequential(
#             nn.Conv2d(in_channels, ch, 5, 1, 2),  # B,  32, 32, 32
#             activation,
#             nn.Conv2d(ch, ch, 5, 2, 2),  # B,  32, 16, 16
#             activation,
#             # Self_Attn(ch, 'relu'),
#         )
#         # self.cnn2 = nn.Sequential(
#         #     nn.Conv2d(ch, ch, 5, 2, 2),  # B,  64,  8,  8
#         #     activation,
#         #     Temporal_Self_Attn(ch, 'relu'),
#         #     nn.Conv2d(ch, ch, 5, 2, 2),  # B,  64,  4,  4
#         #     activation,
#         #     Temporal_Self_Attn(ch, 'relu'),
#         #     nn.Conv2d(ch, ch, 5, 2, 2),  # B, 256,  1,  1
#         #     activation,
#         #     Temporal_Self_Attn(ch, 'relu'),
#         #     # Print_Shape(),
#         #     nn.Conv2d(ch, ch, 5, 2, 2),  # B, 256,  1,  1
#         #     activation,
#         #     nn.Conv2d(ch, n_objects * feat_dim, 5, 2, 2),  # B, 256,  1,  1
#         #     activation
#         # )
#
#         # self.cnn2_0 = nn.Sequential(
#         #     nn.Conv2d(ch, ch, 5, 2, 2),  # B,  64,  8,  8
#         #     activation)
#         #
#         # self.cnn2_1 = nn.Sequential(
#         #     nn.Conv2d(ch, ch, 5, 2, 2),  # B,  64,  4,  4
#         #     activation)
#         # self.t_att_1 = Temporal_Self_Attn(ch, 'relu')
#         #
#         # self.cnn2_2 = nn.Sequential(
#         #     nn.Conv2d(ch, ch, 5, 2, 2),  # B, 256,  1,  1
#         #     activation)
#         #
#         # self.t_att_2 = Temporal_Self_Attn(ch, 'relu')
#         #     # Print_Shape(),
#         #
#         # self.cnn2_3 = nn.Sequential(
#         #     nn.Conv2d(ch, ch, 5, 2, 2),  # B, 256,  1,  1
#         #     activation,
#         #     nn.Conv2d(ch, n_objects * feat_dim, 5, 2, 2),  # B, 256,  1,  1
#         #     activation
#         # )
#
#         # nn.ReplicationPad3d((1, 1, 2, 2, 2, 2))
#         self.cnn3 = nn.Sequential(
#             nn.Conv3d(ch, ch, [3, 5, 5], [1, 2, 2], [1, 2, 2]),
#             # nn.Conv3d(ch, ch, [5, 9, 9], [1, 2, 2], [2, 4, 4]),
#             activation,
#             nn.Conv3d(ch, ch, [3, 5, 5], [1, 2, 2], [1, 2, 2]),
#             activation,
#             Temporal_Self_Attn(ch, 'relu'),
#             nn.Conv3d(ch, ch, [3, 5, 5], [1, 2, 2], [1, 2, 2]),
#             activation,
#             Temporal_Self_Attn(ch, 'relu'),
#             nn.Conv3d(ch, ch, [3, 5, 5], [1, 2, 2], [1, 2, 2]),
#             activation,
#             nn.Conv3d(ch, n_objects * feat_dim, [3, 5, 5], [1, 2, 2], [1, 2, 2]),
#             activation
#         )
#
#         self.linear_pos = nn.Linear(4, ch)
#         self.register_buffer('grid_enc', self._build_grid(self.resolution))
#         self.encoder_pos = self._soft_position_embed
#
#         self.norm = nn.LayerNorm(ch)
#
#         self.mlp = nn.Sequential(nn.Linear(ch, ch),
#                                  activation,
#                                  nn.Linear(ch, ch))
#
#         # self.slot_attention = SlotAttention(n_objects, feat_dim)
#
#     def _soft_position_embed(self, input):
#         return input + self.linear_pos(self.grid_enc)
#
#     def _build_grid(self, resolution):
#         ranges = [np.linspace(0., 1., num=res) for res in resolution]
#         grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
#         grid = np.stack(grid, axis=-1)
#         grid = np.reshape(grid, [resolution[0], resolution[1], -1])
#         grid = np.expand_dims(grid, axis=0)
#         grid = grid.astype(np.float32)
#         return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1)).reshape(grid.shape[0], -1, 4)
#
#     def forward(self, x, reverse=False):
#         bs, T, ch, h, w = x.shape
#
#         x = x.reshape(-1, *x.shape[2:])
#         x = self.cnn(x)
#         x = spatial_flatten_for_sa(x)
#         x = self.encoder_pos(x)
#         x = self.norm(x)
#         x = self.mlp(x) #self.norm(x)
#         # TODO: check out if the norm is fucking something up
#
#         # if reverse:
#         #     x_rev = torch.flip(x.reshape(bs, T, *x.shape[1:]), dims=[1])
#         #     x = x.reshape(bs, T, *x.shape[1:])
#         #     x = torch.stack([x, x_rev], dim=1)
#         #     bs = 2*bs
#         #     x = x.reshape(bs * T, *x.shape[3:])
#
#         # 2D CNNs with temporal attention
#         # x = self.cnn2_0(x.permute(0, 2, 1).reshape(bs*T, x.shape[-1],
#         #                                          int(np.sqrt(x.shape[1])), int(np.sqrt(x.shape[1]))))
#         # x = self.cnn2_1(x)
#         # x = self.t_att_1(x, T)
#         # x = self.cnn2_2(x)
#         # x = self.t_att_2(x, T)
#         # x = self.cnn2_3(x).reshape(bs * self.n_objects * T, self.feat_dim)
#
#         # 3D Conv with or without attention
#         x = x.reshape(bs, T, *x.shape[1:]).permute(0,3,1,2)
#         x = x.reshape(*x.shape[:-1], int(np.sqrt(x.shape[-1])), int(np.sqrt(x.shape[-1])))
#         x = self.cnn3(x)
#         x = x.permute(0, 2, 1, 3, 4)\
#             .reshape(bs, T, self.n_objects, self.feat_dim)\
#             .permute(0, 2, 1, 3).reshape(bs * self.n_objects * T, self.feat_dim)
#
#         return x

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