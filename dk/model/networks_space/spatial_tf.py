import os
import os.path as path
import numpy as np
import torch
import random
import torch.nn as nn

class SpatialTransformation(nn.Module):
    def __init__(self, wh, WH, out_channels, n_objects, zeta_s=0.3, zeta_r=[1, 0.1]): #Note: Changed from 0.1, 1-0.1
        super().__init__()
        self.bg = 0 # No background at this moment
        self.zeta_s, self.zeta_r = zeta_s, zeta_r
        self.w, self.h = wh
        self.W, self.H = WH
        self.out_channels = out_channels
        self.n_objects = n_objects

    def get_sampling_grid(self, y_e, y_p):
        """
        Get sampling grid
        y_e: N * dim_y_e * 1 * 1
        y_p: N * dim_y_p (scale_x, scale_y, trans_x, trans_y)
        """
        # o = self.o

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
        # bb_conf = y_e.data.view(-1, o.dim_y_e)
        # bb_h = h_new.data
        # bb_w = w_new.data
        # bb_center_y = (-trans_y.data + 1)/2 * (o.H - 1) + 1 # [1, H]
        # bb_center_x = (-trans_x.data + 1)/2 * (o.W - 1) + 1 # [1, W]
        # bb_top = bb_center_y - (bb_h-1)/2
        # bb_left = bb_center_x - (bb_w-1)/2
        # bb = torch.cat((bb_left, bb_top, bb_w, bb_h, bb_conf), dim=1) # NTO * 5
        # torch.save(bb.view(-1, o.T, o.O, 5), path.join(o.result_metric_dir, str(self.i)+'.pt'))
        # self.i += 1

        # Generate sampling grid
        # TODO: Generate the inverse trans_mat?
        # TODO: Repeat the channels depending on which feature map we relocate
        grid = nn.functional.affine_grid(trans_mat, torch.Size((trans_mat.size(0), self.out_channels, self.W, self.H))) # N * H * W * 2 , align_corners=False
        return grid, area

    def forward(self, Y_a, Y_s, y_e, grid, use_confi=True):
        bs, ch, w, h = Y_a.shape
        if len(y_e.shape) == 2:
            y_e = y_e[..., None, None]
        Y_s = Y_s.reshape(-1, 1, self.w, self.h) # NTO * 1 * h * w

        # if y_e.mean() < 0.5 and use_confi: # TODO: Independently for each object
        if use_confi:
            Y_a = Y_a * y_e * Y_s # NTO * D * h * w

        # X_s = nn.functional.grid_sample(Y_s, grid) # NTO * 1 * H * W align_corners = False
        X_s = None
        X_a = nn.functional.grid_sample(Y_a, grid) # NTO * D * H * W # locate object in 128x128 layout
        return X_a