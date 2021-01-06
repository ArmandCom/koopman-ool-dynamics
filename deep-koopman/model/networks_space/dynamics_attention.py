from torch import nn
import torch
from functools import reduce
import numpy as np


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
class DynamicsAttention(nn.Module):
    def __init__(self, n_objects, in_dim, dim, last_dim, dyn_dim, feat_map_size, eps = 1e-8, hidden_dim = 64):
        super().__init__()
        self.n_objects = n_objects
        # self.eps = eps
        # self.scale = in_dim ** -0.5

        self.out_dim = dim
        self.mix = nn.Sequential(nn.Conv2d(in_channels=4 + dyn_dim + in_dim, out_channels=in_dim, kernel_size=1),
                                 nn.CELU(),
                                 nn.BatchNorm2d(in_dim))

        self.decoder_initial_size = feat_map_size

        # self.linear_dyn = nn.Linear(dyn_dim, dyn_dim)
        # self.linear_pos = nn.Linear(4, in_dim + dyn_dim)
        self.register_buffer('grid_dec', self._build_grid(self.decoder_initial_size))
        self.decoder_pos = self._soft_position_embed

    def _soft_position_embed(self, input):
        bs = input.shape[0]
        # pos_emb = self.linear_pos(self.grid_dec).permute(0,2,1).reshape(1, -1, *self.decoder_initial_size)
        pos_emb = self.grid_dec.permute(0,2,1).repeat(bs, 1, 1).reshape(bs, -1, *self.decoder_initial_size)
        return torch.cat([input, pos_emb], dim=1)

    def _build_grid(self, resolution):
        ranges = [np.linspace(0., 1., num=res) for res in resolution]
        grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
        grid = np.stack(grid, axis=-1)
        grid = np.reshape(grid, [resolution[0], resolution[1], -1])
        grid = np.expand_dims(grid, axis=0)
        grid = grid.astype(np.float32)
        return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1)).reshape(grid.shape[0], -1, 4)

    def forward(self, inputs, dyn_feat_inputs, ini_slots = None, num_slots = None):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        bs, T, ch, w, h = inputs.size()
        bsOT, dyn_feat = dyn_feat_inputs.size()
        n_obj = bsOT // (bs * T)
        bsO = bs * n_obj
        # print(bs, bs2, h, h2, w, w2, n_obj, T, T2, ch, ch2)

        inputs = inputs[:, None].repeat_interleave(n_obj, dim=1).reshape(bsOT, ch, w, h)

        # We pick the first frame to extract the appearance
        dyn_feats = dyn_feat_inputs[..., None, None].repeat(1, 1, h, w)

        x = self.decoder_pos(torch.cat([inputs, -dyn_feats], dim=1))
        out = self.mix(x)

        return out, None
