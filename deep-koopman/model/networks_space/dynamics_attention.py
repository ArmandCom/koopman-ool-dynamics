from torch import nn
import torch
from functools import reduce
import numpy as np
import random
from utils import util as ut


# Option 1: Attention to individual frames + mean
# TODO: Attention in T*h*w. why? we want to have a weighted sum of the component in all timesteps, so the probabilities should be normalized across time and space and summed.
# class DynamicsAttention(nn.Module):
#     def __init__(self, n_objects, in_dim, in_q_dim, dim, eps = 1e-8, hidden_dim = 64):
#         super().__init__()
#         self.n_objects = n_objects
#         self.eps = eps
#         self.scale = dim ** -0.5
#
#         self.out_dim = dim
#         self.gamma = nn.Parameter(torch.zeros(1))
#
#         # self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
#         # self.slots_sigma = nn.Parameter(torch.randn(1, 1, dim))
#
#         self.to_q = nn.Conv2d(in_channels=in_q_dim, out_channels=in_dim // 2, kernel_size=1)
#         self.to_k = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
#         self.to_v = nn.Conv2d(in_channels=in_dim, out_channels=dim, kernel_size=1)
#
#         # self.gru = nn.GRUCell(dim, dim)
#
#         hidden_dim = max(dim, hidden_dim)
#
#         self.mlp = nn.Sequential(
#             nn.Conv2d(in_channels=dim, out_channels=hidden_dim, kernel_size=1),
#             nn.ReLU(inplace = True),
#             nn.Conv2d(in_channels=hidden_dim, out_channels=dim, kernel_size=1),
#         )
#
#         # self.norm_input  = nn.LayerNorm(dim)
#         # self.norm_slots  = nn.LayerNorm(dim)
#         # self.norm_pre_ff = nn.LayerNorm(dim)
#
#     def forward(self, inputs, dyn_feat_inputs, ini_slots = None, num_slots = None):
#         """
#             inputs :
#                 x : input feature maps( B X C X W X H)
#             returns :
#                 out : self attention value + input feature
#                 attention: B X N X N (N is Width*Height)
#         """
#
#         bs, T, ch, w, h = inputs.size()
#         bs2, n_obj, T2, ch2, w2, h2 = dyn_feat_inputs.size()
#         bsTO = bs * n_obj * T
#         # print(bs, bs2, h, h2, w, w2, n_obj, T, T2, ch, ch2)
#         inputs = inputs[:, None].repeat(1, n_obj, 1, 1, 1, 1).reshape(bsTO, ch, w, h)
#         dyn_feat_inputs = dyn_feat_inputs.reshape(bsTO, ch2, w, h)
#
#         # Query, Key
#         # print(dyn_feat_inputs.shape, inputs.shape)
#         q = self.to_q(dyn_feat_inputs).view(bsTO, -1, w * h).permute(0, 2, 1)  # B X C X (W*H)
#         k = self.to_k(inputs).view(bsTO, -1, w * h)  # B X C x (W*H)
#
#         # Compute energy
#         energy = torch.bmm(q, k) # * self.scale # transpose check, Why scale?
#         # Original slot attention
#         # energy_obj = energy.reshape(bs, n_obj, T, *energy.shape[1:])
#         # attn_obj = energy_obj.softmax(dim=1) + self.eps
#
#         # Self-attention
#         attn_frame = energy.softmax(dim=-1) # + self.eps
#
#         # We multiply by the union of both objects, as we don't want to amplify the 0ed regions.
#         # attn = attn_obj * attn_frame.sum(dim=1, keepdim=True)
#         # attn = attn / attn.sum(dim=-1, keepdim=True)
#
#         attn = attn_frame
#
#         # Get value and output
#         v = self.to_v(inputs).reshape(bsTO, -1, w * h)  # B X C X N
#         print(v.shape)
#         out = torch.bmm(v, attn.permute(0, 2, 1))
#         print(out.shape)
#         out = out.view(bsTO, -1, w, h)
#
#         # out = self.mlp(out) + out
#         # out = self.gamma * out + inputs
#
#         return out, attn

# Option 2: Mask across time and slot attention
#  - We first mask + sum across time to get the initial slots. Then we perform slot attention.
# class DynamicsAttention(nn.Module):
#     def __init__(self, n_objects, in_dim, in_q_dim, dim, eps = 1e-8, hidden_dim = 64):
#         super().__init__()
#         self.n_objects = n_objects
#         self.eps = eps
#         self.scale = dim ** -0.5
#
#         self.out_dim = dim
#         self.gamma = nn.Parameter(torch.zeros(1))
#
#         # self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
#         # self.slots_sigma = nn.Parameter(torch.randn(1, 1, dim))
#
#         self.to_q = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
#         self.to_k = nn.Conv3d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
#         self.to_v = nn.Conv3d(in_channels=in_dim, out_channels=dim, kernel_size=1)
#
#         # self.gru = nn.GRUCell(dim, dim)
#
#         hidden_dim = max(dim, hidden_dim)
#
#         self.mlp = nn.Sequential(
#             nn.Conv2d(in_channels=dim, out_channels=hidden_dim, kernel_size=1),
#             nn.ReLU(inplace = True),
#             nn.Conv2d(in_channels=hidden_dim, out_channels=dim, kernel_size=1),
#         )
#
#         # self.norm_input  = nn.LayerNorm(dim)
#         # self.norm_slots  = nn.LayerNorm(dim)
#         # self.norm_pre_ff = nn.LayerNorm(dim)
#
#     def forward(self, inputs, dyn_feat_inputs, ini_slots = None, num_slots = None):
#         """
#             inputs :
#                 x : input feature maps( B X C X W X H)
#             returns :
#                 out : self attention value + input feature
#                 attention: B X N X N (N is Width*Height)
#         """
#
#         bs, T, ch, w, h = inputs.size()
#         bs2, n_obj, T2, ch2, w2, h2 = dyn_feat_inputs.size()
#         bsTO = bs * n_obj * T
#         bsO = bs * n_obj
#         # print(bs, bs2, h, h2, w, w2, n_obj, T, T2, ch, ch2)
#         inputs = inputs[:, None].repeat(1, n_obj, 1, 1, 1, 1)
#
#         logits_inputs = inputs.permute(0, 1, 3, 2, 4, 5).reshape(bsO, ch, T, w, h)
#         inputs = inputs.reshape(bsO, T, ch, w, h)
#         dyn_feat_inputs = dyn_feat_inputs.reshape(bsO, T, ch2, w, h)
#
#         dyn_feat_attn = dyn_feat_inputs.softmax(1)
#         logits_query = (inputs * dyn_feat_attn).sum(1)
#         # inputs = inputs.reshape
#
#         # Query, Key
#         # print(dyn_feat_inputs.shape, inputs.shape)
#         q = self.to_q(logits_query).view(bsO, -1, w * h).permute(0, 2, 1)  # B X C X (W*H)
#         k = self.to_k(logits_inputs).view(bsO, -1, T * w * h)  # B X C x (W*H)
#
#         # Compute energy
#         energy = torch.bmm(q, k) # * self.scale # transpose check, Why scale?
#         # Original slot attention
#         # energy_obj = energy.reshape(bs, n_obj, T, *energy.shape[1:])
#         # attn_obj = energy_obj.softmax(dim=1) + self.eps
#
#         # Self-attention
#         attn_frame = energy.softmax(dim=-1) # + self.eps
#
#         # We multiply by the union of both objects, as we don't want to amplify the 0ed regions.
#         # attn = attn_obj * attn_frame.sum(dim=1, keepdim=True)
#         # attn = attn / attn.sum(dim=-1, keepdim=True)
#
#         attn = attn_frame
#
#         # Get value and output
#         v = self.to_v(logits_inputs).reshape(bsO, -1, T * w * h)  # B X C X N
#         out = torch.bmm(v, attn.permute(0, 2, 1))
#         out = out.view(bsO, -1, w, h)
#
#         # out = self.mlp(out) + out
#         # out = self.gamma * out + inputs
#
#         return out, attn

# Option 3: Full attention across time
# class DynamicsAttention(nn.Module):
#     def __init__(self, n_objects, in_dim, in_q_dim, dim, eps = 1e-8, hidden_dim = 64):
#         super().__init__()
#         self.n_objects = n_objects
#         self.eps = eps
#         self.scale = dim ** -0.5
#
#         self.out_dim = dim
#         self.gamma = nn.Parameter(torch.zeros(1))
#
#         # self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
#         # self.slots_sigma = nn.Parameter(torch.randn(1, 1, dim))
#
#         self.to_q = nn.Conv3d(in_channels=in_q_dim, out_channels=in_dim // 2, kernel_size=1)
#         self.to_k = nn.Conv3d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
#         self.to_v = nn.Conv3d(in_channels=in_dim, out_channels=dim, kernel_size=1)
#
#         # self.gru = nn.GRUCell(dim, dim)
#
#         hidden_dim = max(dim, hidden_dim)
#
#         self.mlp = nn.Sequential(
#             nn.Conv2d(in_channels=dim, out_channels=hidden_dim, kernel_size=1),
#             nn.ReLU(inplace = True),
#             nn.Conv2d(in_channels=hidden_dim, out_channels=dim, kernel_size=1),
#         )
#
#         # self.norm_input  = nn.LayerNorm(dim)
#         # self.norm_slots  = nn.LayerNorm(dim)
#         # self.norm_pre_ff = nn.LayerNorm(dim)
#
#     def forward(self, inputs, dyn_feat_inputs, ini_slots = None, num_slots = None):
#         """
#             inputs :
#                 x : input feature maps( B X C X W X H)
#             returns :
#                 out : self attention value + input feature
#                 attention: B X N X N (N is Width*Height)
#         """
#
#         bs, T, ch, w, h = inputs.size()
#         bs2, n_obj, T2, ch2, w2, h2 = dyn_feat_inputs.size()
#         bsTO = bs * n_obj * T
#         bsO = bs * n_obj
#         # print(bs, bs2, h, h2, w, w2, n_obj, T, T2, ch, ch2)
#         inputs = inputs[:, None].repeat(1, n_obj, 1, 1, 1, 1).permute(0,1,3,2,4,5).reshape(bsO, ch, T, w, h)
#         dyn_feat_inputs = dyn_feat_inputs.permute(0,1,3,2,4,5).reshape(bsO, ch2, T, w, h)
#
#         # Query, Key
#         # print(dyn_feat_inputs.shape, inputs.shape)
#         print(dyn_feat_inputs.shape)
#         q = self.to_q(dyn_feat_inputs).reshape(bsO, -1, T * w * h)[..., 10:].permute(0, 2, 1)  # B X C X (W*H)
#         k = self.to_k(inputs).reshape(bsO, -1, T * w * h)  # B X C x (W*H)
#
#         # Compute energy
#         energy = torch.bmm(q, k) # * self.scale # transpose check, Why scale?
#         # Original slot attention
#         # energy_obj = energy.reshape(bs, n_obj, T, *energy.shape[1:])
#         # attn_obj = energy_obj.softmax(dim=1) + self.eps
#
#         # Self-attention
#         attn_frame = energy.softmax(dim=-1) # + self.eps
#
#         # We multiply by the union of both objects, as we don't want to amplify the 0ed regions.
#         # attn = attn_obj * attn_frame.sum(dim=1, keepdim=True)
#         # attn = attn / attn.sum(dim=-1, keepdim=True)
#
#         attn = attn_frame
#         print(q.shape, k.shape, attn_frame.shape)
#         # Get value and output
#         v = self.to_v(inputs).view(bsO, -1, T * w * h)
#
#         out = torch.bmm(v, attn.permute(0, 2, 1))
#
#         out = out.view(bsTO, -1, w, h)
#
#         # out = self.mlp(out) + out
#         # out = self.gamma * out + inputs
#
#         return out, attn

# Option 3: First attention, then Temporal weighting
# class DynamicsAttention(nn.Module):
#     def __init__(self, n_objects, in_dim, dim, last_dim, dyn_dim, input_dim, eps = 1e-8, hidden_dim = 64):
#         super().__init__()
#         self.n_objects = n_objects
#         self.eps = eps
#         self.scale = last_dim ** -0.5
#
#         self.out_dim = dim
#         self.gamma = nn.Parameter(torch.zeros(1))
#
#         # self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
#         # self.slots_sigma = nn.Parameter(torch.randn(1, 1, dim))
#
#         self.to_q = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
#         self.to_k = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
#         self.to_v = nn.Conv2d(in_channels=in_dim, out_channels=dim, kernel_size=1)
#
#         # Temporal attention
#         hw = reduce((lambda x, y: x * y), input_dim)
#         self.to_tq = nn.Linear(last_dim, last_dim)
#         self.to_tk = nn.Linear(last_dim, last_dim)
#         self.to_tv = nn.Linear(last_dim, last_dim)
#
#         self.slots_mu = nn.Parameter(torch.randn(1, 1, last_dim))
#         self.slots_sigma = nn.Parameter(torch.randn(1, 1, last_dim))
#
#         # self.gru = nn.GRUCell(dim, dim)
#
#         hidden_dim = max(last_dim, hidden_dim)
#
#         self.last_mlp = nn.Sequential(
#             nn.Linear(last_dim, hidden_dim),
#             nn.ReLU(inplace = True),
#             nn.Linear(hidden_dim, last_dim),
#         )
#
#         # self.norm_input  = nn.LayerNorm(dim)
#         # self.norm_slots  = nn.LayerNorm(dim)
#         # self.norm_pre_ff = nn.LayerNorm(dim)
#
#     def forward(self, inputs, dyn_feat_inputs, ini_slots = None, num_slots = None):
#         """
#             inputs :
#                 x : input feature maps( B X C X W X H)
#             returns :
#                 out : self attention value + input feature
#                 attention: B X N X N (N is Width*Height)
#         """
#
#         bs, T, ch, w, h = inputs.size()
#         bs2, n_obj, T2, ch2, w2, h2 = dyn_feat_inputs.size()
#         bsTO = bs * n_obj * T
#         bsO = bs * n_obj
#         # print(bs, bs2, h, h2, w, w2, n_obj, T, T2, ch, ch2)
#         logits_inputs = inputs[:, None].repeat(1, n_obj, 1, 1, 1, 1).reshape(bsTO, ch, w, h)
#         logits_query = dyn_feat_inputs.reshape(bsTO, ch2, w, h)
#
#         # inputs = inputs.reshape
#
#         # Query, Key
#         # print(dyn_feat_inputs.shape, inputs.shape)
#         q = self.to_q(logits_query).view(bsTO, -1, w * h).permute(0, 2, 1)  # B X C X (W*H)
#         k = self.to_k(logits_inputs).view(bsTO, -1, w * h)  # B X C x (W*H)
#
#         # Compute energy
#         energy = torch.bmm(q, k) # * self.scale # transpose check, Why scale?
#         # Original slot attention
#         # energy_obj = energy.reshape(bs, n_obj, T, *energy.shape[1:])
#         # attn_obj = energy_obj.softmax(dim=1) + self.eps
#
#         # Spatial Self-attention
#         # energy = energy.reshape(bsO, T, *energy.shape[1:])
#         attn = energy.softmax(dim=-1) # + self.eps
#
#         # Get value and output
#         v = self.to_v(logits_inputs).reshape(bsTO, -1, w * h)  # B X C X N
#         out = torch.bmm(v, attn.permute(0, 2, 1)).view(bsTO, -1, w, h)
#
#         # out = self.gamma * out + inputs
#
#         return out, attn
#
#     def forward_last(self, inputs, n_iters=3):
#         bs, n_obj, T, feat_dim = inputs.shape
#         inputs = inputs.reshape(bs * n_obj, T, feat_dim)
#
#         # out = (inputs.softmax(1) * inputs).sum(1)[:, None]
#
#         mu = self.slots_mu.expand(bs, 1, -1)
#         sigma = self.slots_sigma.expand(bs, 1, -1)
#         tq = torch.normal(mu, sigma)
#
#
#         tk = self.to_tk(inputs)
#         tv = self.to_tv(inputs)
#
#         for _ in range(n_iters):
#
#             tq = self.to_tq(tq)
#             dots = torch.einsum('bid,bjd->bij', tq, tk) * self.scale
#             attn = dots.softmax(dim=1) + self.eps
#             attn = attn / attn.sum(dim=-1, keepdim=True)
#             updates = torch.einsum('bjd,bij->bid', tv, attn)
#             tq = updates + self.last_mlp(updates)
#
#         return tq

# Option 4: First attention, then Temporal weighting
# class DynamicsAttention(nn.Module):
#     def __init__(self, n_objects, in_dim, dim, last_dim, dyn_dim, input_dim, eps = 1e-8, hidden_dim = 64):
#         super().__init__()
#         self.n_objects = n_objects
#         self.eps = eps
#         self.scale = in_dim ** -0.5
#
#         self.out_dim = dim
#         self.gamma = nn.Parameter(torch.zeros(1))
#
#         # self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
#         # self.slots_sigma = nn.Parameter(torch.randn(1, 1, dim))
#
#         self.to_q = nn.Conv2d(in_channels=in_dim, out_channels=1, kernel_size=1)
#         self.to_k = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
#         self.to_v = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
#
#     def forward(self, inputs, dyn_feat_inputs, ini_slots = None, num_slots = None):
#         """
#             inputs :
#                 x : input feature maps( B X C X W X H)
#             returns :
#                 out : self attention value + input feature
#                 attention: B X N X N (N is Width*Height)
#         """
#
#         bs, T, ch, w, h = inputs.size()
#         bs2, n_obj, T2, ch2, w2, h2 = dyn_feat_inputs.size()
#         bsO = bs * n_obj
#         # print(bs, bs2, h, h2, w, w2, n_obj, T, T2, ch, ch2)
#
#         # We pick the first frame to extract the appearance
#         logits_inputs = inputs[:, 0].reshape(bsO, ch, w, h)
#         logits_query = dyn_feat_inputs[:, :, 0].reshape(bsO, ch2, w, h)
#
#         # Query, Key
#         # print(dyn_feat_inputs.shape, inputs.shape)
#         q = self.to_q(logits_query).view(bsO, -1, w * h).reshape(bs, n_obj, w * h) # B X C X (W*H)
#         k = self.to_k(logits_inputs).view(bs, -1, w * h)  # B X C x (W*H)
#         v = self.to_k(logits_inputs).view(bs, -1, w * h)
#
#         dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
#         attn = dots.softmax(dim=1) + self.eps
#         attn = attn / attn.sum(dim=-1, keepdim=True)
#
#         objs = torch.einsum('bjd,bij->bid', v, attn)
#         attn_obj = objs.reshape(bsO, 1, h*w).softmax(dim=-1).reshape(bsO, 1, h, w)
#         out = logits_inputs * attn_obj
#         # out, attn = logits_inputs, None
#
#         return out, attn

# Option 5: use dynamic features to locate it.
# class DynamicsAttention(nn.Module):
#     def __init__(self, n_objects, in_dim, dim, last_dim, dyn_dim, feat_map_size, eps = 1e-8, hidden_dim = 64):
#         super().__init__()
#         self.n_objects = n_objects
#         # self.eps = eps
#         # self.scale = in_dim ** -0.5
#
#         self.out_dim = dim
#         self.mix = nn.Sequential(nn.Conv2d(in_channels=4 + dyn_dim + in_dim, out_channels=in_dim, kernel_size=1),
#                                  nn.CELU(),
#                                  nn.BatchNorm2d(in_dim))
#
#         self.decoder_initial_size = feat_map_size
#
#         # self.linear_dyn = nn.Linear(dyn_dim, dyn_dim)
#         # self.linear_pos = nn.Linear(4, in_dim + dyn_dim)
#         self.register_buffer('grid_dec', self._build_grid(self.decoder_initial_size))
#         self.decoder_pos = self._soft_position_embed
#
#     def _soft_position_embed(self, input):
#         bs = input.shape[0]
#         # pos_emb = self.linear_pos(self.grid_dec).permute(0,2,1).reshape(1, -1, *self.decoder_initial_size)
#         pos_emb = self.grid_dec.permute(0,2,1).repeat(bs, 1, 1).reshape(bs, -1, *self.decoder_initial_size)
#         return torch.cat([input, pos_emb], dim=1)
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
#     def forward(self, inputs, dyn_feat_inputs, ini_slots = None, num_slots = None):
#         """
#             inputs :
#                 x : input feature maps( B X C X W X H)
#             returns :
#                 out : self attention value + input feature
#                 attention: B X N X N (N is Width*Height)
#         """
#         bs, T, ch, w, h = inputs.size()
#         bsOT, dyn_feat = dyn_feat_inputs.size()
#         n_obj = bsOT // (bs * T)
#         bsO = bs * n_obj
#         # print(bs, bs2, h, h2, w, w2, n_obj, T, T2, ch, ch2)
#
#         inputs = inputs[:, None].repeat_interleave(n_obj, dim=1).reshape(bsOT, ch, w, h)
#
#         # We pick the first frame to extract the appearance
#         dyn_feats = dyn_feat_inputs[..., None, None].repeat(1, 1, h, w)
#
#         x = self.decoder_pos(torch.cat([inputs, -dyn_feats], dim=1))
#         out = self.mix(x)
#
#         return out, None

# Option 7: as slot attention- LATEST
# class DynamicsAttention(nn.Module):
#     def __init__(self, in_dim, out_dim, padding, resolution, eps = 1e-8):
#         super().__init__()
#         self.eps = eps
#         self.scale = in_dim ** -0.5
#
#         # self.gamma = nn.Parameter(torch.zeros(1))
#
#         self.padding = padding
#
#         # self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
#         # self.slots_sigma = nn.Parameter(torch.randn(1, 1, dim))
#
#         self.to_q = nn.Conv2d        (in_channels=in_dim,     out_channels=in_dim // 2, kernel_size=1)
#         #
#         # self.to_k_for_q   = nn.Conv2d(in_channels=in_dim,     out_channels=in_dim // 2, kernel_size=1)
#         self.to_k = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
#         #
#         # self.to_v_for_q   = nn.Conv2d(in_channels=in_dim,     out_channels=in_dim,      kernel_size=1)
#         self.to_v = nn.Conv2d(in_channels=in_dim + 4, out_channels=in_dim,      kernel_size=1)
#         # self.to_v_ini = nn.Conv2d(in_channels=in_dim + 4, out_channels=in_dim,      kernel_size=1)
#
#         self.mlp = nn.Linear(in_dim, out_dim)
#         self.register_buffer('pos_enc', self._build_grid(resolution))
#
#     def forward(self, inputs):
#         """
#             inputs :
#                 x : input feature maps( B X C X W X H)
#             returns :
#                 out : self attention value + input feature
#                 attention: B X N X N (N is Width*Height)
#         """
#
#         bs, T, ch, w, h = inputs.size()
#
#         inputs_pe = torch.cat([inputs, self.pos_enc.repeat(bs, T, 1, 1, 1)], dim=-3).reshape(bs*T, -1, w, h)
#         vs = self.to_v(inputs_pe).reshape(bs, T, -1, w * h).transpose(1, 0) # TODO: Add temporal encoding before v?
#         ks = self.to_k(inputs.reshape(bs*T, -1, w, h)).reshape(bs, T, -1, w * h).transpose(1, 0)
#         input_q = inputs[:, 0]
#
#         # zeros = torch.zeros_like(inputs[:,:1]).repeat(1, self.padding, 1, 1, 1)
#         # inputs = torch.cat([inputs, zeros], dim=1)
#         # inputs = inputs.reshape(bs*(T+self.padding), -1, w, h)
#         # v_qs = self.to_v_for_q(inputs).reshape(bs, T + self.padding, -1, w * h).transpose(1, 0)
#         # k_qs = self.to_k_for_q(inputs).reshape(bs, T + self.padding, -1, w * h).transpose(1, 0)
#
#         frames = []
#         attns = []
#         # mask = torch.zeros_like(input_q[:, 0]).reshape(bs, h * w)
#         for t in range(T):
#             # TODO: Compute self-attention to the first frame?
#             #  If it doesn't work, try slot-encoder method. Each value gets assigned to a single query.
#             # TODO: Idea. Do an iterative method for finding dyn by showing the koopman embedding to the dyn encoder
#             # We pick the first frame to extract the appearance
#             if t == 0:
#                 # frames.append(vs[t].reshape(bs, -1, h*w))
#                 q = self.to_q(input_q).reshape(bs, -1, h * w).permute(0, 2, 1)
#                 # ones_idx = torch.argmax(input_q.reshape(bs, -1, h * w).sum(1), dim=-1, keepdim=True)
#                 # mask[ones_idx] = 1
#                 # mask = mask[:, :, None, None]
#
#             v = vs[t].transpose(2, 1).reshape(bs, -1, h*w)
#             k = ks[t].transpose(2, 1).reshape(bs, -1, h*w)
#
#             # TODO: compute attention to the second and third separately.
#             #  Then keep part of the second and sum the second and third to compute the next query.
#
#             # dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
#             # attn = dots.softmax(dim=1) + self.eps
#             # attn = attn / attn.sum(dim=-1, keepdim=True)
#             # objs = torch.einsum('bjd,bij->bid', v, attn)
#
#             energy = torch.bmm(q, k) * self.scale # transpose check, Why scale?
#             attn = energy.softmax(dim=1) + self.eps
#             attn = attn / attn.sum(dim=-1, keepdim=True)
#             # attn = energy.softmax(dim=-1) + self.eps
#             out = torch.bmm(v, attn.permute(0, 2, 1))
#
#             # q = self.to_q(out.reshape(bs, -1, w, h)).reshape(bs, -1, w * h).permute(0, 2, 1)
#
#             attns.append(attn)
#             frames.append(out)
#
#         attns = torch.stack(attns, dim=1).permute(0, 3, 1, 2) # bs, w*h, T, f_out
#         frames = self.mlp(torch.stack(frames, dim=1).permute(0, 3, 1, 2)) # bs, w*h, T, f_out
#         # if random.random() < 0.0005:
#         #     for i in range (w*h):
#         #         print(frames[0, i].norm())
#         return frames, attns
#
#     def _build_grid(self, resolution):
#         ranges = [np.linspace(0., 1., num=res) for res in resolution]
#         grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
#         grid = np.reshape(grid, [2, resolution[0], resolution[1]])
#         grid = np.expand_dims(grid, axis=[0,1])
#         grid = grid.astype(np.float32)
#         return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=2))

# Option 6: Number of objects is a grid in space. Tracking occurs by soft attention
# class DynamicsAttention(nn.Module):
#     def __init__(self, in_dim, out_dim, padding, resolution, eps = 1e-8):
#         super().__init__()
#         self.eps = eps
#         self.scale = in_dim ** -0.5
#
#         # self.gamma = nn.Parameter(torch.zeros(1))
#
#         self.padding = padding
#
#         # self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
#         # self.slots_sigma = nn.Parameter(torch.randn(1, 1, dim))
#
#         self.to_q = nn.Conv2d        (in_channels=in_dim,     out_channels=in_dim // 2, kernel_size=1)
#         #
#         self.to_k_for_q   = nn.Conv2d(in_channels=in_dim,     out_channels=in_dim // 2, kernel_size=1)
#         self.to_k_for_out = nn.Conv2d(in_channels=in_dim + 4, out_channels=in_dim // 2, kernel_size=1)
#         #
#         self.to_v_for_q   = nn.Conv2d(in_channels=in_dim,     out_channels=in_dim,      kernel_size=1)
#         self.to_v_for_out = nn.Conv2d(in_channels=in_dim + 4, out_channels=in_dim,      kernel_size=1)
#
#         self.mlp = nn.Linear(in_dim, out_dim)
#         self.register_buffer('pos_enc', self._build_grid(resolution))
#
#     def forward(self, inputs):
#         """
#             inputs :
#                 x : input feature maps( B X C X W X H)
#             returns :
#                 out : self attention value + input feature
#                 attention: B X N X N (N is Width*Height)
#         """
#
#         bs, T, ch, w, h = inputs.size()
#
#         n_obj = w*h
#         # print(bs, bs2, h, h2, w, w2, n_obj, T, T2, ch, ch2)
#
#         inputs_pe = torch.cat([inputs, self.pos_enc.repeat(bs, T, 1, 1, 1)], dim=-3).reshape(bs*T, -1, w, h)
#         v_outs = self.to_v_for_out(inputs_pe).reshape(bs, T, -1, w * h).transpose(1, 0) # TODO: Add temporal encoding before v?
#         k_outs = self.to_k_for_out(inputs_pe).reshape(bs, T, -1, w * h).transpose(1, 0)
#
#         zeros = torch.zeros_like(inputs[:,:1]).repeat(1, self.padding, 1, 1, 1)
#         inputs = torch.cat([inputs, zeros], dim=1)
#         inputs = inputs.reshape(bs*(T+self.padding), -1, w, h)
#
#         v_qs = self.to_v_for_q(inputs).reshape(bs, T + self.padding, -1, w * h).transpose(1, 0)
#         k_qs = self.to_k_for_q(inputs).reshape(bs, T + self.padding, -1, w * h).transpose(1, 0)
#
#         frames = []
#         attns = []
#         for t in range(T):
#             # TODO: Compute self-attention to the first frame?
#             #  If it doesn't work, try slot-encoder method. Each value gets assigned to a single query.
#             # TODO: Idea. Do an iterative method for finding dyn by showing the koopman embedding to the dyn encoder
#             # We pick the first frame to extract the appearance
#             if t == 0:
#
#                 frames.append(v_outs[t].reshape(bs, -1, h*w))
#                 # q = self.to_q(v_qs[t].reshape(bs, -1, h, w)).reshape(bs, -1, h * w).permute(0, 2, 1)
#                 q = self.to_q(v_outs[t].reshape(bs, -1, h, w)).reshape(bs, -1, h * w).permute(0, 2, 1)
#             else:
#                 # TODO: Not done properly, query has to be computed at each temporal step from the previous output.
#
#                 v_q =     v_qs[t: t+1+self.padding].transpose(2, 1).reshape(bs, -1, h*w*(1+self.padding))
#                 v_out = v_outs[t]                  .transpose(2, 1).reshape(bs, -1, h*w)
#
#                 k_q =     k_qs[t: t+1+self.padding].transpose(2, 1).reshape(bs, -1, h*w*(1+self.padding))
#                 k_out = k_outs[t]                  .transpose(2, 1).reshape(bs, -1, h*w)
#
#                 # TODO: compute attention to the second and third separately.
#                 #  Then keep part of the second and sum the second and third to compute the next query.
#
#                 # dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
#                 # attn = dots.softmax(dim=1) + self.eps
#                 # attn = attn / attn.sum(dim=-1, keepdim=True)
#                 # objs = torch.einsum('bjd,bij->bid', v, attn)
#
#                 energy_out = torch.bmm(q, k_out) #* self.scale # transpose check, Why scale?
#                 # Spatial Self-attention
#                 attn_out = energy_out.softmax(dim=-1) + self.eps
#                 out = torch.bmm(v_out, attn_out.permute(0, 2, 1))
#                 # print(out.shape, attn_out.shape)
#                 frames.append(out)
#
#                 q = self.to_q(out.reshape(bs, -1, w, h)).reshape(bs, -1, w * h).permute(0, 2, 1)
#
#                 # energy_q = torch.bmm(q, k_q) #* self.scale
#                 # attn_q = energy_q.softmax(dim=-1) + self.eps
#                 # q_logit = torch.bmm(v_q, attn_q.permute(0, 2, 1)).view(bs, -1, w, h)
#                 # # print(q_logit.shape, attn_q.shape)
#                 # q = self.to_q(q_logit).reshape(bs, -1, w * h).permute(0, 2, 1)
#
#                 attns.append(attn_out)
#
#         attns = torch.stack(attns, dim=1).permute(0, 3, 1, 2) # bs, w*h, T, f_out
#         frames = self.mlp(torch.stack(frames, dim=1).permute(0, 3, 1, 2)) # bs, w*h, T, f_out
#         return frames, attns
#
#     def _build_grid(self, resolution):
#         ranges = [np.linspace(0., 1., num=res) for res in resolution]
#         grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
#         grid = np.reshape(grid, [2, resolution[0], resolution[1]])
#         grid = np.expand_dims(grid, axis=[0,1])
#         grid = grid.astype(np.float32)
#         return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=2))


# Option 7: From Tracking by Animation: Unsupervised Learning of Multi-Object Attentive Trackers
# class DynamicsAttention(nn.Module):
#     def __init__(self, in_dim, out_dim, padding, resolution, eps = 1e-8):
#         super().__init__()
#         self.eps = eps
#         self.scale = in_dim ** -0.5
#
#         # self.gamma = nn.Parameter(torch.zeros(1))
#
#         self.q_size = (4, 4)
#         self.n_objects = self.q_size[0]*self.q_size[1]
#
#         self.padding = padding
#
#         # self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
#         # self.slots_sigma = nn.Parameter(torch.randn(1, 1, dim))
#
#         # We assume an initial C size of: 16x16
#         self.to_ini_q = nn.Sequential(nn.Conv2d(in_channels=in_dim, out_channels=(in_dim // 2), kernel_size=4, stride=2, padding=1), # 8x8
#                                       nn.Conv2d(in_channels=(in_dim // 2), out_channels=(in_dim // 2), kernel_size=4, stride=2, padding=1), # 4x4
#                                       nn.Conv2d(in_channels=(in_dim // 2), out_channels=in_dim, kernel_size=4, stride=2, padding=1), # 2x2
#                                       nn.Conv2d(in_channels=in_dim, out_channels=(in_dim // 2) * self.n_objects, kernel_size=4, stride=2, padding=1), # 1x1
#                                       )
#
#         self.to_q = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
#         self.to_k = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
#         self.to_v = nn.Conv2d(in_channels=in_dim, out_channels=in_dim,      kernel_size=1)
#         self.to_beta = nn.Conv2d(in_channels=in_dim, out_channels=1,        kernel_size=1)
#
#         self.soft_pos_enc = nn.Conv2d(in_channels=4, out_channels=in_dim,   kernel_size=1)
#
#         self.mlp = nn.Linear(in_dim, out_dim)
#         self.register_buffer('pos_enc', self._build_grid(resolution))
#
#         self.cosine_similarity = nn.CosineSimilarity(dim=2)
#         self.softmax = nn.Softmax(dim=1)
#         self.rnn_cell = nn.GRUCell(in_dim, in_dim)
#
#     def forward(self, inputs):
#         """
#             inputs :
#                 x : input feature maps( B X C X W X H)
#             returns :
#                 out : self attention value + input feature
#                 attention: B X N X N (N is Width*Height)
#         """
#
#         bs, T, ch, w, h = inputs.size()
#
#         input_q = inputs[:, 0]
#
#         inputs = inputs.reshape(bs*T, -1, w, h)
#         # inputs_pe = torch.cat([inputs, self.pos_enc.repeat(bs, T, 1, 1, 1)], dim=-3).reshape(bs*T, -1, w, h)
#         inputs_pe =  inputs + self.soft_pos_enc(self.pos_enc)
#
#         vs = self.to_v(inputs_pe).reshape(bs, T, -1, w * h).transpose(1, 0)
#         ks = self.to_k(inputs   ).reshape(bs, T, -1, w * h).transpose(1, 0)
#         betas = self.to_beta(inputs).reshape(bs, T, -1, w * h).transpose(1, 0)
#
#         frames = []
#         attns = []
#
#         for t in range(T):
#
#             if t == 0:
#                 q = self.to_ini_q(input_q).reshape(bs, -1, self.n_objects).permute(0, 2, 1)
#
#             v = vs[t].reshape(bs, -1, h * w)
#             k = ks[t].reshape(bs, -1, h * w)
#             # Betas
#             beta_pre = betas[t].reshape(bs, -1, h * w)
#             beta_pos = beta_pre.clamp(min=0)
#             beta_neg = beta_pre.clamp(max=0)
#             beta = beta_neg.exp().log1p() + beta_pos + (-beta_pos).exp().log1p() + (1 - np.log(2)) # N * 1
#
#             energy = torch.bmm(q, k) #* self.scale # transpose check, Why scale?
#             # attn = energy.softmax(dim=1) + self.eps
#             # attn = attn / attn.sum(dim=-1, keepdim=True)
#             attn = energy.softmax(dim=-1) * beta + self.eps
#             r = torch.bmm(v, attn.permute(0, 2, 1))
#             r = r.permute(0, 2, 1).reshape(bs * self.n_objects, -1)
#
#             # New tracker state. Parallell for all obj.
#             if t == 0:
#                 prev_out = torch.zeros_like(r)
#             out = self.rnn_cell(r, prev_out)
#             prev_out = out
#             out = out.reshape(bs, self.n_objects, -1).permute(0, 2, 1)
#
#             # next query
#             q = self.to_q(out.reshape(bs, -1, self.q_size[0], self.q_size[1]))\
#                 .reshape(bs, -1, self.n_objects).permute(0, 2, 1)
#
#             attns.append(attn)
#             frames.append(out)
#
#         attns = torch.stack(attns, dim=1).permute(0, 3, 1, 2) # bs, w*h, T, f_out
#         frames = self.mlp(torch.stack(frames, dim=1).permute(0, 3, 1, 2)) # bs, w*h, T, f_out
#         # if random.random() < 0.0005:
#         #     for i in range (w*h):
#         #         print(frames[0, i].norm())
#
#         '''
#         ------------------------------------------------------------------------------------
#         '''
#         # # Addressing key
#         # k = self.linear_k(h_o_prev) # N * C2_2
#         # k_expand = k.unsqueeze(1).expand_as(C) # N * C2_1 * C2_2
#         # # Key strength, which equals to beta_pre.exp().log1p() + 1 but avoids 'inf' caused by exp()
#         # beta_pre = self.linear_b(h_o_prev)
#         # beta_pos = beta_pre.clamp(min=0)
#         # beta_neg = beta_pre.clamp(max=0)
#         # beta = beta_neg.exp().log1p() + beta_pos + (-beta_pos).exp().log1p() + (1 - np.log(2)) # N * 1
#         # # Weighting
#         # C_cos = smd.Identity()(C)
#         # smd.norm_grad(C_cos, 1)
#         # s = self.cosine_similarity(C_cos, k_expand).view(-1, o.dim_C2_1) # N * C2_1
#         # w = self.softmax(s * beta) # N * C2_1
#         #
#         # # Read vector
#         # w1 = w.unsqueeze(1) # N * 1 * C2_1
#         # smd.norm_grad(w1, 1)
#         # r = w1.bmm(C).squeeze(1) # N * C2_2
#         # # RNN
#         # h_o = self.rnn_cell(r, h_o_prev)
#         #
#         # if "no_mem" not in o.exp_config:
#         #     # Erase vector
#         #     e = self.linear_e(h_o).sigmoid().unsqueeze(1) # N * 1 * C2_2
#         #     # Write vector
#         #     v = self.linear_v(h_o).unsqueeze(1) # N * 1 * C2_2
#         #     # Update memory
#         #     w2 = w.unsqueeze(2) # N * C2_1 * 1
#         #     C = C * (1 - w2.bmm(e)) + w2.bmm(v) # N * C2_1 * C2_2
#
#
#         return frames, attns
#
#     def _build_grid(self, resolution):
#         ranges = [np.linspace(0., 1., num=res) for res in resolution]
#         grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
#         grid = np.reshape(grid, [2, resolution[0], resolution[1]])
#         grid = np.expand_dims(grid, axis=[0])
#         grid = grid.astype(np.float32)
#         return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=1))

# Option 8: atention like the paper
class DynamicsAttention(nn.Module):
    def __init__(self, in_dim, out_dim, padding, resolution, eps = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = in_dim ** -0.5

        # self.gamma = nn.Parameter(torch.zeros(1))

        self.q_size = (4, 4)
        self.n_objects = self.q_size[0]*self.q_size[1]

        self.padding = padding

        # self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        # self.slots_sigma = nn.Parameter(torch.randn(1, 1, dim))

        # We assume an initial C size of: 16x16
        self.to_ini_q = nn.Sequential(nn.Conv2d(in_channels=in_dim, out_channels=(in_dim // 2), kernel_size=4, stride=2, padding=1), # 8x8
                                      nn.Conv2d(in_channels=(in_dim // 2), out_channels=(in_dim // 2), kernel_size=4, stride=2, padding=1), # 4x4
                                      nn.Conv2d(in_channels=(in_dim // 2), out_channels=in_dim, kernel_size=4, stride=2, padding=1), # 2x2
                                      nn.Conv2d(in_channels=in_dim, out_channels=(in_dim // 2) * self.n_objects, kernel_size=4, stride=2, padding=1), # 1x1
                                      )

        self.to_q = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
        self.to_k = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
        self.to_v = nn.Conv2d(in_channels=in_dim, out_channels=in_dim,      kernel_size=1)
        self.to_beta = nn.Conv2d(in_channels=in_dim, out_channels=1,        kernel_size=1)

        self.soft_pos_enc = nn.Conv2d(in_channels=4, out_channels=in_dim,   kernel_size=1)

        self.mlp = nn.Linear(in_dim, out_dim)
        self.register_buffer('pos_enc', self._build_grid(resolution))

        self.cosine_similarity = nn.CosineSimilarity(dim=2)
        self.softmax = nn.Softmax(dim=1)
        self.rnn_cell = nn.GRUCell(in_dim, in_dim)

    def forward(self, inputs):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """

        bs, T, ch, w, h = inputs.size()

        input_q = inputs[:, 0]

        inputs = inputs.reshape(bs*T, -1, w, h)
        # inputs_pe = torch.cat([inputs, self.pos_enc.repeat(bs, T, 1, 1, 1)], dim=-3).reshape(bs*T, -1, w, h)
        inputs_pe =  inputs + self.soft_pos_enc(self.pos_enc)

        vs = self.to_v(inputs_pe).reshape(bs, T, -1, w * h).transpose(1, 0)
        ks = self.to_k(inputs   ).reshape(bs, T, -1, w * h).transpose(1, 0)
        betas = self.to_beta(inputs).reshape(bs, T, -1, w * h).transpose(1, 0)

        frames = []
        attns = []

        for t in range(T):

            if t == 0:
                q = self.to_ini_q(input_q).reshape(bs, -1, self.n_objects).permute(0, 2, 1)

            v = vs[t].reshape(bs, -1, h * w)
            k = ks[t].reshape(bs, -1, h * w)
            # Betas
            beta_pre = betas[t].reshape(bs, -1, h * w)
            beta_pos = beta_pre.clamp(min=0)
            beta_neg = beta_pre.clamp(max=0)
            beta = beta_neg.exp().log1p() + beta_pos + (-beta_pos).exp().log1p() + (1 - np.log(2)) # N * 1

            energy = torch.bmm(q, k) #* self.scale # transpose check, Why scale?
            # attn = energy.softmax(dim=1) + self.eps
            # attn = attn / attn.sum(dim=-1, keepdim=True)
            attn = energy.softmax(dim=-1) * beta + self.eps
            r = torch.bmm(v, attn.permute(0, 2, 1))
            r = r.permute(0, 2, 1).reshape(bs * self.n_objects, -1)

            # New tracker state. Parallell for all obj.
            if t == 0:
                prev_out = torch.zeros_like(r)
            out = self.rnn_cell(r, prev_out)
            prev_out = out
            out = out.reshape(bs, self.n_objects, -1).permute(0, 2, 1)

            # next query
            q = self.to_q(out.reshape(bs, -1, self.q_size[0], self.q_size[1])) \
                .reshape(bs, -1, self.n_objects).permute(0, 2, 1)

            attns.append(attn)
            frames.append(out)

        attns = torch.stack(attns, dim=1).permute(0, 3, 1, 2) # bs, w*h, T, f_out
        frames = self.mlp(torch.stack(frames, dim=1).permute(0, 3, 1, 2)) # bs, w*h, T, f_out
        # if random.random() < 0.0005:
        #     for i in range (w*h):
        #         print(frames[0, i].norm())

        '''
        ------------------------------------------------------------------------------------
        '''
        # # Addressing key
        # k = self.linear_k(h_o_prev) # N * C2_2
        # k_expand = k.unsqueeze(1).expand_as(C) # N * C2_1 * C2_2
        # # Key strength, which equals to beta_pre.exp().log1p() + 1 but avoids 'inf' caused by exp()
        # beta_pre = self.linear_b(h_o_prev)
        # beta_pos = beta_pre.clamp(min=0)
        # beta_neg = beta_pre.clamp(max=0)
        # beta = beta_neg.exp().log1p() + beta_pos + (-beta_pos).exp().log1p() + (1 - np.log(2)) # N * 1
        # # Weighting
        # C_cos = smd.Identity()(C)
        # smd.norm_grad(C_cos, 1)
        # s = self.cosine_similarity(C_cos, k_expand).view(-1, o.dim_C2_1) # N * C2_1
        # w = self.softmax(s * beta) # N * C2_1
        #
        # # Read vector
        # w1 = w.unsqueeze(1) # N * 1 * C2_1
        # smd.norm_grad(w1, 1)
        # r = w1.bmm(C).squeeze(1) # N * C2_2
        # # RNN
        # h_o = self.rnn_cell(r, h_o_prev)
        #
        # if "no_mem" not in o.exp_config:
        #     # Erase vector
        #     e = self.linear_e(h_o).sigmoid().unsqueeze(1) # N * 1 * C2_2
        #     # Write vector
        #     v = self.linear_v(h_o).unsqueeze(1) # N * 1 * C2_2
        #     # Update memory
        #     w2 = w.unsqueeze(2) # N * C2_1 * 1
        #     C = C * (1 - w2.bmm(e)) + w2.bmm(v) # N * C2_1 * C2_2


        return frames, attns

    def _build_grid(self, resolution):
        ranges = [np.linspace(0., 1., num=res) for res in resolution]
        grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
        grid = np.reshape(grid, [2, resolution[0], resolution[1]])
        grid = np.expand_dims(grid, axis=[0])
        grid = grid.astype(np.float32)
        return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=1))