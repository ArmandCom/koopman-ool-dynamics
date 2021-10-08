import os
import os.path as path
import numpy as np
import torch
import torch.nn as nn

from model.networks_space.background import BackgroundAE

class SpatialTransformation(nn.Module):

    def __init__(self, hw, HW, out_channels, n_objects, zeta_s=0.3, zeta_r=[1, 0.1]): #Note: Changed from 0.1, 1-0.1
        super().__init__()
        self.bg = 0 # No background at this moment
        self.zeta_s, self.zeta_r = zeta_s, zeta_r
        self.h, self.w = hw
        self.H, self.W = HW
        self.out_channels = out_channels
        self.bgNet = BackgroundAE(out_channels)
        self.n_objects = n_objects

    def forward(self, y_e, y_l, Y_s, Y_a, grid, **kwargs):
                # (self, Y_a, Y_s, y_e, grid, use_confi=True)

        B, O, T, l_dim = y_l.shape

        Y_b = kwargs['Y_b'].view(-1, self.out_channels, self.H, self.W) if 'Y_b' in kwargs.keys() else None

        # TODO: Background generation code

        # if o.task == 'mnist':
        #     Y_s = Y_s.data.clone().fill_(1)

        # TODO: Modify appearance to be constant (at least partly).
        #  We can add a deterministic image decoder in the renderer.

        # Spatial transform
        Y_s = Y_s.reshape(-1, 1, self.h, self.w) * y_e[..., None, None] # NOT * 1 * h * w
        Y_a = Y_a.reshape(-1, self.out_channels, self.h, self.w) * Y_s # NOT * D * h * w
        X_s = nn.functional.grid_sample(torch.ones_like(Y_s), grid, align_corners=False) # NOT * 1 * H * W
        X_a = nn.functional.grid_sample(Y_a, grid, align_corners=False) # NOT * D * H * W

        # Permute, and generate layers
        X_s = X_s.view(B, O, T, 1 * self.H * self.W).transpose(1, 2).flatten(0, 1) # NT * O * 1HW
        X_a = X_a.view(B, O, T, self.out_channels * self.H * self.W).transpose(1, 2).flatten(0, 1) # NT * O * DHW
        y_l = y_l.view(B, O, T, l_dim) # NT * O * dim_y_l
        y_l = y_l.permute(0, 2, 3, 1).flatten(0, 1) # NT * dim_y_l * O
        X_s = y_l.bmm(X_s).clamp(max=1) # NT * dim_y_l * 1HW
        X_a = y_l.bmm(X_a) # NT * dim_y_l * DHW
        # Note: if o.task == 'mnist':
        X_a = X_a.clamp(max=1)

        # Y_b = None # Note: test.
        if Y_b is not None:
            X_s_acc_neg = 1 - X_s.sum(1).clamp(max=1)\
                .reshape(-1, 1, self.H, self.W)
            Y_b = self.bgNet((Y_b * X_s_acc_neg).reshape(-1, T, self.out_channels, self.H, self.W))

        # Reconstruct iteratively
        X_s_split = torch.unbind(X_s.reshape(-1, l_dim, 1, self.H, self.W), 1)  # NT * 1 * H * W
        X_a_split = torch.unbind(X_a.reshape(-1, l_dim, self.out_channels, self.H, self.W), 1) # NT * D * H * W
        X_r = Y_b.repeat_interleave(T, dim=1).flatten(0, 1) if Y_b is not None else X_a_split[0].data.clone().zero_() # NT * D * H * W
        for i in range(0, l_dim):
            # X_r = X_r + X_s_split[i] * (X_a_split[i] - X_r)
            X_r = X_r * (1 - X_s_split[i]) + X_a_split[i]

        X_r = X_r.view(B, T, self.out_channels, self.H, self.W) # N * T * D * H * W

        return X_r


    def get_sampling_grid(self, y_e, y_p):
        """
        y_e: N * dim_y_e * 1 * 1
        y_p: N * dim_y_p (scale_x, scale_y, trans_x, trans_y)
        """

        # Generate 2D transformation matrix
        if len(y_e.shape) == 2:
            y_e = y_e[..., None, None]

        trans_x, trans_y, scale, ratio = y_p.split(1, 1) # N * 1
        scale = 0.6 + self.zeta_s*scale
        ratio = self.zeta_r[0] + self.zeta_r[1]*ratio
        ratio_sqrt = ratio.sqrt()
        area = scale * scale
        h_new = self.h * scale * ratio_sqrt
        w_new = self.w * scale / ratio_sqrt
        scale_x = self.W / w_new
        scale_y = self.H / h_new
        if self.bg == 0:
            trans_x = (1 - (self.w*2/3)/self.W) * trans_x
            trans_y = (1 - (self.h*2/3)/self.H) * trans_y
        zero = trans_x.data.clone().zero_() # N * 1
        trans_mat = torch.cat((scale_x, zero, scale_x * trans_x, zero, scale_y,
                               scale_y * trans_y), 1).view(-1, 2, 3) # N * 2 * 3

        # Convert to bounding boxes and save
        # if o.metric == 1 and o.v == 0:
        #     bb_conf = y_e.data.view(-1, o.dim_y_e)
        #     bb_h = h_new.data
        #     bb_w = w_new.data
        #     bb_center_y = (-trans_y.data + 1)/2 * (self.H - 1) + 1 # [1, H]
        #     bb_center_x = (-trans_x.data + 1)/2 * (self.W - 1) + 1 # [1, W]
        #     bb_top = bb_center_y - (bb_h-1)/2
        #     bb_left = bb_center_x - (bb_w-1)/2
        #     bb = torch.cat((bb_left, bb_top, bb_w, bb_h, bb_conf), dim=1) # NTO * 5
        #     # torch.save(bb.view(-1, o.T, o.n_objects, 5), path.join(o.result_metric_dir, str(self.i)+'.pt'))
        #     self.i += 1

        # Generate sampling grid
        grid = nn.functional.affine_grid(trans_mat, torch.Size((trans_mat.size(0), self.out_channels, self.H, self.W)), align_corners=False) # N * H * W * 2
        return grid, area
