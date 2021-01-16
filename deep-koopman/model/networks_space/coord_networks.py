import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

class CoordEncoder(nn.Module):

    def __init__(self, feat_cte_dim, feat_dyn_dim, feat_map_size):
        super(CoordEncoder, self).__init__()

        # self.n_objects = n_objects
        # self.feat_cte_dim = feat_cte_dim
        # TODO:
        #  - r: [0,1](Sigmoid?), angle: [0, 2pi] or (x,y,s)
        #  - rot: [0, pi](clamp), sign: [-1,1](Tanh)


        self.resolution = feat_map_size
        self.feat_dyn_dim = feat_dyn_dim
        self.feat_cte_warp_dim = feat_cte_dim #// 4

        hidden_dim = 128

        self.motion_nw = MotionNetwork(self.feat_cte_warp_dim, self.feat_dyn_dim, hidden_dim, pos_enc=True)

        self.grid = Grid()
        self.register_buffer('base_grid', self.grid._build_base_grid(self.resolution))
        self.register_buffer('pos_enc', self.grid._build_grid(self.resolution))

    def _warp(self, input, motion_field):
        # N, n_channels, object_size, object_size
        obj = F.grid_sample(input, motion_field)
        return obj

    def forward(self, x, dyn_feat_input=None, block='backbone'):
        input_shape = x.shape
        bs, T, ch, h, w = input_shape
        bsOT, dyn_feat = dyn_feat_input.shape
        dyn_feat_input = dyn_feat_input.reshape(bs, -1, T, dyn_feat, 1, 1).repeat(1, 1, 1, 1, h, w) # 32, 2, 8, 7, 16, 16

        x = x.unsqueeze(1).repeat_interleave(dyn_feat_input.shape[1], dim=1) # 32, 2, 8, 32, 16, 16
        input = torch.cat([x[:, :, :, :self.feat_cte_warp_dim], dyn_feat_input], dim=-3).reshape(bsOT, self.feat_cte_warp_dim + self.feat_dyn_dim, h, w)
        input_pe = torch.cat([input, self.pos_enc.repeat(bsOT, 1, 1, 1)], dim=1) # 512, 44, 16, 16

        motion = self.motion_nw(input_pe, act='clamp')

        motion_field = (self.base_grid + motion).permute(0, 2, 3, 1) # 512, 16, 16, 2
        out = self._warp(x.reshape(-1, ch, h, w), motion_field) # 512, 32, 16, 16

        return out, motion

class CoordDecoder(nn.Module):

    def __init__(self, init_dim, dyn_dim, feat_map_size):
        super(CoordDecoder, self).__init__()

        # self.n_objects = n_objects
        # self.feat_cte_dim = feat_cte_dim

        self.resolution = feat_map_size

        hidden_dim = 128

        self.motion_nw = MotionNetwork(init_dim, dyn_dim, hidden_dim, pos_enc=False)

        self.grid = Grid()
        self.register_buffer('base_grid', self.grid._build_base_grid(self.resolution))
        # self.register_buffer('pos_enc', self.grid._build_grid(self.resolution))

    def _warp(self, input, motion_field):
        # N, n_channels, object_size, object_size
        obj = F.grid_sample(input, motion_field, mode='nearest')
        return obj

    def forward(self, x, dyn_feat_input=None):
        input_shape = x.shape
        bsOT, ch, h, w = input_shape
        if dyn_feat_input is not None:
            x = torch.cat([x, dyn_feat_input.repeat(1, 1, h, w)], dim=1)
        motion = self.motion_nw(x, act='clamp')
        # TODO: Motion_nw with differentiated parameters (theta, x, y, s1, s2). Perd la gracia?

        motion_field = (self.base_grid + motion).permute(0, 2, 3, 1) # 512, 16, 16, 2
        out = self._warp(x.reshape(-1, ch, h, w), motion_field) # 512, 32, 16, 16

        return out, motion

class MaskEncoder(nn.Module):

    def __init__(self, feat_cte_dim, feat_dyn_dim, feat_map_size):
        super(MaskEncoder, self).__init__()

        # self.n_objects = n_objects
        # self.feat_cte_dim = feat_cte_dim
        # TODO:
        #  - r: [0,1](Sigmoid?), angle: [0, 2pi] or (x,y,s)
        #  - rot: [0, pi](clamp), sign: [-1,1](Tanh)


        self.resolution = feat_map_size
        self.feat_dyn_dim = feat_dyn_dim
        self.feat_cte_dim = feat_cte_dim

        hidden_dim = 128

        self.motion_nw = MotionNetwork(self.feat_cte_warp_dim, self.feat_dyn_dim, hidden_dim, pos_enc=True)

        self.grid = Grid()
        self.register_buffer('base_grid', self.grid._build_base_grid(self.resolution))
        self.register_buffer('pos_enc', self.grid._build_grid(self.resolution))

    def _warp(self, input, motion_field):
        # N, n_channels, object_size, object_size
        obj = F.grid_sample(input, motion_field)
        return obj

    def forward(self, x, dyn_feat_input=None, block='backbone'):
        input_shape = x.shape
        bs, T, ch, h, w = input_shape
        bsOT, dyn_feat = dyn_feat_input.shape
        dyn_feat_input = dyn_feat_input.reshape(bs, -1, T, dyn_feat, 1, 1).repeat(1, 1, 1, 1, h, w) # 32, 2, 8, 7, 16, 16

        x = x.unsqueeze(1).repeat_interleave(dyn_feat_input.shape[1], dim=1) # 32, 2, 8, 32, 16, 16
        input = torch.cat([x[:, :, :, :self.feat_cte_dim], dyn_feat_input], dim=-3).reshape(bsOT, self.feat_cte_dim + self.feat_dyn_dim, h, w)
        input_pe = torch.cat([input, self.pos_enc.repeat(bsOT, 1, 1, 1)], dim=1) # 512, 44, 16, 16

        mask = self.motion_nw(input_pe, act='clamp')

        return mask

# --------------------------------------------------------------------------

class MotionNetwork(nn.Module):
    def __init__(self, feat_cte_dim, feat_dyn_dim, hidden_dim, pos_enc=False):
        super(MotionNetwork, self).__init__()
        n_coords = 4 if pos_enc else 0
        self.nn = nn.Sequential(
            nn.Conv2d(n_coords + feat_cte_dim + feat_dyn_dim, hidden_dim, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(hidden_dim),
            nn.Conv2d(hidden_dim, 32, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 2, 1),
        )
    def forward(self, x, act=None):
        out = self.nn(x)

        alpha = 0.1
        if act == 'tanh':
            out = F.tanh(alpha * out)
        elif act == 'clamp':
            out = out.clamp(-1, 1)
        elif act is None:
            pass
        else:
            print('!! Not implemented')
            exit()
        return out

# class MotionStructNetwork(nn.Module):
#     def __init__(self, feat_cte_dim, feat_dyn_dim, hidden_dim, pos_enc=False, mode='xys'):
#         super(MotionStructNetwork, self).__init__()
#         n_coords = 4 if pos_enc else 0
#         self.mode = mode
#
#         layers = [nn.Conv2d(n_coords + feat_cte_dim + feat_dyn_dim, hidden_dim, 3, 2, 1), nn.CELU(),
#                   nn.BatchNorm2d(hidden_dim), nn.Conv2d(hidden_dim, hidden_dim, 3, 2, 1), nn.CELU(),
#                   nn.BatchNorm2d(hidden_dim), nn.Conv2d(hidden_dim, hidden_dim, 3, 2, 1), nn.CELU(),
#                   nn.BatchNorm2d(hidden_dim), nn.Conv2d(hidden_dim, 32, 3, 2, 1), nn.CELU(), nn.BatchNorm2d(32)
#                   ]
#
#         if mode == 'xys':
#             self.max_scale = 3
#             layers.append(nn.Conv2d(32, 3, 1))
#
#         self.nn = nn.Sequential(*layers)
#
#     def forward(self, x, act=None):
#         out = self.nn(x).squeeze(-1).squeeze(-1)
#
#         if self.mode=='xys':
#             x, y, s = out.clamp(-1, 1).chunk(3, dim=-1)
#             s = 1 + ((s+1)/2) * self.max_scale
#
#         mf = ... # TODO: Transform into 2 channel matrix of size (resolution)
#
#         return mf

# TODO: Mask the number uniquely.
class MaskNetwork(nn.Module):
    def __init__(self, in_dim, feat_dyn_dim, hidden_dim, n_objects, pos_enc=False):
        super(MaskNetwork, self).__init__()
        self.softmax = nn.Softmax(1) # Select object slot
        self.temp = 1
        n_coords = 4 if pos_enc else 0
        n_slots = n_objects # TODO: +1 if background

        self.nn = nn.Sequential(
            nn.Conv2d(n_coords + in_dim + feat_dyn_dim, hidden_dim, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(hidden_dim),
            nn.Conv2d(hidden_dim, 32, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, n_slots, 1),
        )

    def forward(self, x, act=None):
        out = self.softmax(self.nn(x).squeeze(-1).squeeze(-1) / self.temp)
        return out

class KeyPointsExtractor(nn.Module):

    def __init__(self, feat_dyn_dim, hidden_dim, n_points, resolution=None, pos_enc=True):
        super(KeyPointsExtractor, self).__init__()
        self.temp = 1
        n_coords = 2 if pos_enc else 0

        if resolution is not None:
            self.resolution = resolution
        else:
            self.resolution = [64, 64]

        self.nn = nn.Sequential(
            nn.Conv2d(n_coords + feat_dyn_dim, hidden_dim, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(hidden_dim),
            nn.Conv2d(hidden_dim, n_points, 1),
        )

        self.softmax2d = nn.Softmax2d()

        self.grids = Grid()
        self.register_buffer('grid', self.grids._build_warping_grid(self.resolution))
        # self.register_buffer('grid_softargmax', self.grids._build_softmax_grid(self.resolution))

    def forward(self, x):

        bsOT, dyn_feat = x.shape
        dyn_feat_input = x[..., None, None].repeat(1, 1, *self.resolution) # 32 * 2 * 8, 7, 16, 16
        pos_enc = self.grid.repeat(bsOT, 1, 1, 1) # 32 * 2 * 8, 7, 16, 16
        x_pe = torch.cat([dyn_feat_input, pos_enc], dim=1) # 512, 7 + 2, 16, 16

        sm_out = self.softmax2d(self.nn(x_pe) / self.temp)

        argsm_out_map = sm_out.unsqueeze(2) * pos_enc.unsqueeze(1)
        argsm_out = argsm_out_map.sum(-1).sum(-1)
        argsm_out = torch.clamp(argsm_out, -1, 1)


        # print(argsm_out_map.max(), argsm_out.max(), argsm_out.min())


        # fig, axs = plt.subplots(1, 2, figsize=(23, 10))
        # axs = axs.ravel()
        #
        # axs[0].set_title('image source')
        # im = axs[0].imshow(sm_out[0].sum(-3).squeeze().cpu().detach())
        # fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.5)
        #
        # # axs[1].axis('off')
        # # axs[1].set_title('image destination pytorch')
        # # axs[1].imshow(image_warp[0].squeeze().cpu())
        # #
        # # axs[2].axis('off')
        # # axs[2].set_title('image destination')
        # # axs[2].imshow(image_ori_warp[0].squeeze().cpu())
        #
        # plt.savefig('heatmap_test.png')
        # print('kornia test done.')
        # exit()


        return sm_out, argsm_out


class Grid():
    def __init__(self):
        super(Grid, self).__init__()

    def _build_grid(self, resolution):
        ranges = [np.linspace(0., 1., num=res) for res in resolution]
        grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
        grid = np.stack(grid, axis=-1)
        grid = np.reshape(grid, [-1, resolution[0], resolution[1]])
        grid = np.expand_dims(grid, axis=0)
        grid = grid.astype(np.float32)
        return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=1))

    def _build_warping_grid(self, resolution):
        ranges = [np.linspace(-1., 1., num=res) for res in resolution]
        grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
        grid = np.stack(grid, axis=-1)
        grid = np.reshape(grid, [-1, resolution[0], resolution[1]])
        grid = np.expand_dims(grid, axis=0)
        grid = grid.astype(np.float32)
        # torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=1))
        return torch.from_numpy(grid)

    def _build_softmax_grid(self, resolution):
        ranges = [np.linspace(0, 1., num=res) for res in resolution]
        grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
        grid = np.stack(grid, axis=-1)
        grid = np.reshape(grid, [-1, resolution[0], resolution[1]])
        grid = np.expand_dims(grid, axis=0)
        grid = grid.astype(np.float32)
        # orch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=1))
        return torch.from_numpy(grid)

    def _build_base_grid(self, resolution):
        ranges = [np.linspace(-1., 1., num=res) for res in resolution]
        grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
        grid = np.stack(grid, axis=-1)
        grid = np.reshape(grid, [-1, resolution[0], resolution[1]])
        grid = np.expand_dims(grid, axis=0)
        grid = grid.astype(np.float32)
        return torch.from_numpy(grid)

# def spatial_flatten_for_sa(x):
#   x = x.permute(0,2,3,1)
#   return x.reshape(-1, x.shape[1] * x.shape[2], x.shape[-1])