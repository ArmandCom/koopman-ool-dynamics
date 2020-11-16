import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class ImageEncoder(nn.Module):

    def __init__(self, in_channels, feat_dim, n_objects, ngf, n_layers):
        super(ImageEncoder, self).__init__()

        self.n_objects = n_objects
        self.feat_dim = feat_dim
        ch = 32
        self.resolution = (32, 32)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, ch, 5, 1, 2),  # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(ch, ch, 5, 2, 2),  # B,  32, 16, 16
            nn.ReLU(True))

        self.cnn2 = nn.Sequential(
            nn.Conv2d(ch, ch, 5, 2, 2),  # B,  64,  8,  8
            nn.ReLU(True),
            nn.Conv2d(ch, ch, 5, 2, 2),  # B,  64,  4,  4
            nn.ReLU(True),
            nn.Conv2d(ch, ch, 5, 2, 2),  # B, 256,  1,  1
            nn.ReLU(True),
            nn.Conv2d(ch, ch, 5, 2, 2),  # B, 256,  1,  1
            nn.ReLU(True),
            nn.Conv2d(ch, n_objects * feat_dim, 5, 2, 2),  # B, 256,  1,  1
            nn.ReLU(True)
        )
        # nn.ReplicationPad3d((1, 1, 2, 2, 2, 2))
        self.cnn3 = nn.Sequential(
            nn.Conv3d(ch, ch, [3, 5, 5], [1, 2, 2], [1, 2, 2]),
            nn.ReLU(True),
            nn.Conv3d(ch, ch, [3, 5, 5], [1, 2, 2], [1, 2, 2]),
            nn.ReLU(True),
            nn.Conv3d(ch, ch, [3, 5, 5], [1, 2, 2], [1, 2, 2]),
            nn.ReLU(True),
            nn.Conv3d(ch, ch, [3, 5, 5], [1, 2, 2], [1, 2, 2]),
            nn.ReLU(True),
            nn.Conv3d(ch, n_objects * feat_dim, [3, 5, 5], [1, 2, 2], [1, 2, 2]),
            nn.ReLU(True)
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
                                 nn.ReLU(True),
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

        if reverse:
            x_rev = torch.flip(x.reshape(bs, T, *x.shape[1:]), dims=[1])
            x = x.reshape(bs, T, *x.shape[1:])
            x = torch.stack([x, x_rev], dim=1)
            bs = 2*bs
            x = x.reshape(bs * T, *x.shape[3:])

        x = x.reshape(bs, T, *x.shape[1:]).permute(0,3,1,2)
        x = x.reshape(*x.shape[:-1], int(np.sqrt(x.shape[-1])), int(np.sqrt(x.shape[-1])))
        x = self.cnn3(x)
        x = x.permute(0, 2, 1, 3, 4)\
            .reshape(bs, T, self.n_objects, self.feat_dim)\
            .permute(0, 2, 1, 3).reshape(bs * self.n_objects * T, self.feat_dim)

        return x

# class AttImageEncoder(nn.Module):
#
#     def __init__(self, in_channels, feat_dim, n_objects, ngf, n_layers):
#         super(AttImageEncoder, self).__init__()
#
#         ch = 32
#         self.resolution = (64, 64)
#         self.cnn = nn.Sequential(
#             nn.Conv2d(in_channels, ch, 5, 1, 2),          # B,  32, 32, 32
#             nn.ReLU(True),
#             nn.Conv2d(ch, ch, 5, 1, 2),          # B,  32, 16, 16
#             nn.ReLU(True),
#             nn.Conv2d(ch, ch, 5, 1, 2),          # B,  64,  8,  8
#             nn.ReLU(True),
#             nn.Conv2d(ch, ch, 5, 1, 2),          # B,  64,  4,  4
#             nn.ReLU(True),
#             nn.Conv2d(ch, feat_dim, 5, 1, 2),            # B, 256,  1,  1
#             nn.ReLU(True),
#         )
#
#         # self.cnn3 = nn.Sequential(
#         #     nn.Conv3d(in_channels, 32, [3, 5, 5], [1, 1, 1], [1, 2, 2]),
#         #     nn.ReLU(True),
#         #     nn.Conv3d(32, 32, [3, 5, 5], [1, 1, 1], [1, 2, 2]),
#         #     nn.ReLU(True),
#         #     nn.Conv3d(32, 32, [3, 5, 5], [1, 1, 1], [1, 2, 2]),
#         #     nn.ReLU(True),
#         #     nn.Conv3d(32, 32, [3, 5, 5], [1, 1, 1], [1, 2, 2]),
#         #     nn.ReLU(True),
#         #     nn.Conv3d(32, feat_dim, [3, 5, 5], [1, 1, 1], [1, 2, 2]),
#         #     nn.ReLU(True),
#         # )
#         # x = x.permute(0, 2, 1, 3, 4)
#         # x = self.cnn3(x)
#         # x = x.permute(0, 2, 1, 3, 4)
#         # x = x.reshape(-1, *x.shape[2:])
#
#         self.linear_pos = nn.Linear(4, feat_dim)
#         self.register_buffer('grid_enc', self._build_grid(self.resolution))
#         self.encoder_pos = self._soft_position_embed
#
#         self.norm = nn.LayerNorm(feat_dim)
#
#         self.mlp = nn.Sequential(nn.Linear(feat_dim, feat_dim),
#                                  nn.ReLU(True),
#                                  nn.Linear(feat_dim, feat_dim))
#
#
#
#         # self.slot_attention = SlotAttention(n_objects, feat_dim)
#
#     def _soft_position_embed(self, input):
#
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
#     def forward(self, x):
#         bs, T, ch, h, w = x.shape
#
#         x = x.reshape(-1, *x.shape[2:])
#         x = self.cnn(x)
#         x = spatial_flatten_for_sa(x)
#         x = self.encoder_pos(x)
#         x = self.mlp(self.norm(x))
#
#         # Option 1: time-aware object segmentation
#         x = x.reshape(bs, T, *x.shape[1:])
#         x_t = []
#         for t in range(T):
#             n_iters = 3 if t==0 else 2
#             x_t.append(self.slot_attention(x[:,t], n_iters))
#         x = torch.stack(x_t, dim=1)
#         x = x.reshape(-1, *x.shape[2:])
#
#         # Option 2: regular object segmentation
#         # x = self.slot_attention(x, 3)
#
#         return x

def spatial_flatten_for_sa(x):
  x = x.permute(0,2,3,1)
  return x.reshape(-1, x.shape[1] * x.shape[2], x.shape[-1])

# class SlotAttention(nn.Module):
#     def __init__(self, num_slots, dim, iters=3, eps=1e-8, hidden_dim=64):
#         super().__init__()
#
#         self.num_slots = num_slots
#         self.iters = iters
#         self.eps = eps
#         self.scale = dim ** -0.5
#
#         self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
#         self.slots_sigma = nn.Parameter(torch.randn(1, 1, dim))
#
#         self.to_q = nn.Linear(dim, dim)
#         self.to_k = nn.Linear(dim, dim)
#         self.to_v = nn.Linear(dim, dim)
#
#         self.gru = nn.GRUCell(dim, dim)
#
#         hidden_dim = max(dim, hidden_dim)
#
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden_dim, dim)
#         )
#
#         self.norm_input = nn.LayerNorm(dim)
#         self.norm_slots = nn.LayerNorm(dim)
#         self.norm_pre_ff = nn.LayerNorm(dim)
#
#     def forward(self, inputs, iters=3, num_slots=None):
#         b, n, d = inputs.shape
#         n_s = num_slots if num_slots is not None else self.num_slots
#
#         mu = self.slots_mu.expand(b, n_s, -1)
#         sigma = self.slots_sigma.expand(b, n_s, -1)
#         slots = torch.normal(mu, sigma)
#
#         inputs = self.norm_input(inputs)
#         k, v = self.to_k(inputs), self.to_v(inputs)
#
#         for _ in range(iters):
#             slots_prev = slots
#
#             slots = self.norm_slots(slots)
#             q = self.to_q(slots)
#
#             dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
#             attn = dots.softmax(dim=1) + self.eps
#             attn = attn / attn.sum(dim=-1, keepdim=True)
#
#             updates = torch.einsum('bjd,bij->bid', v, attn)
#
#             slots = self.gru(
#                 updates.reshape(-1, d),
#                 slots_prev.reshape(-1, d)
#             )
#
#             slots = slots.reshape(b, -1, d)
#             slots = slots + self.mlp(self.norm_pre_ff(slots))
#
#         return slots