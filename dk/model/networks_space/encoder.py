import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model.networks_space.tracking import TrackerArray
import numpy as np
from functools import reduce


class ImageEncoder(nn.Module):

    def __init__(self, in_channels, feat_cte_dim, feat_dyn_dim, resolution, n_objects, ngf, image_size, cte_app=False, bs=40):
        super(ImageEncoder, self).__init__()

        self.feat_dyn_dim = feat_dyn_dim
        self.feat_cte_dim = feat_cte_dim
        # self.n_objects = reduce((lambda x, y: x * y), resolution)
        self.n_objects = n_objects

        self.ori_resolution = image_size
        self.cte_resolution = (32, 32)

        last_hidden_dim = 96

        layers = [nn.Conv2d(in_channels + 4, 16, 4, 2, 1), # 64, 64
                  nn.CELU(), # use ReLU
                  nn.BatchNorm2d(16), # not use batchnorm
                  nn.Conv2d(16, 32, 3, 1, 1),
                  nn.CELU(),
                  nn.BatchNorm2d(32),
                  nn.Conv2d(32, 64, 4, 2, 1), # 32, 32
                  nn.CELU(),
                  nn.BatchNorm2d(64),
                  nn.Conv2d(64, 64, 3, 1, 1),
                  nn.CELU(),
                  nn.BatchNorm2d(64),
                  nn.Conv2d(64, 128, 4, 2, 1), # 16, 16
                  nn.CELU(),
                  nn.BatchNorm2d(128),
                  nn.Conv2d(128, 128, 3, 1, 1),
                  nn.CELU(),
                  nn.BatchNorm2d(128)]

        if image_size[0] == 128:
            layers.extend([nn.Conv2d(128, 128, 4, 2, 1), # 8, 8
                            nn.CELU(),
                            nn.BatchNorm2d(128),
                            nn.Conv2d(128, 256, 3, 1, 1),
                            nn.CELU(),
                            nn.BatchNorm2d(256), # TODO: Bias to False.
                            nn.Conv2d(256, last_hidden_dim, 1)])
        elif image_size[0] == 64:
            layers.extend([nn.Conv2d(128, 256, 3, 1, 1),
                           nn.CELU(),
                           nn.BatchNorm2d(256), # TODO: Bias to False.
                           nn.Conv2d(256, last_hidden_dim, 1)])
        else:
            raise NotImplementedError
        self.cnn_attn = nn.Sequential(*layers)
        # SPACE: lateral encoder

        '''Tracker params'''
        dims = {'confidence': 1, # confidence dim is always 1
                'layer': 1,
                'pose': feat_dyn_dim, # Dymensions of the pose = 4
                'shape': 1, #self.cte_resolution[0]*self.cte_resolution[1],
                'appearance':feat_cte_dim, # size of appearance vector
                'tracker_state':last_hidden_dim * 3, # hidden state of the tracker
                'features':last_hidden_dim} # feature channels

        self.dim_h_o = dims['tracker_state']
        self.dim_y_e = dims['confidence']
        self.tracker = TrackerArray(self.n_objects, dims)

        # States
        self.states = {}
        self.states['h_o_prev'] = torch.Tensor(1, self.n_objects, self.dim_h_o) # TODO: Save as parameter. We can save one of the parameters and replicate
        self.states['y_e_prev'] = torch.Tensor(1, self.n_objects, self.dim_y_e)
        self.reset_states()

        # # Update parameters
        # self.state_params = {}
        # self.state_params['h_o_prev'] = nn.Parameter(torch.zeros(1, self.n_objects, self.dim_h_o), requires_grad=False)
        # self.state_params['y_e_prev'] = nn.Parameter(torch.zeros(1, self.n_objects, self.dim_y_e), requires_grad=False)

        # self.linear_pos = nn.Linear(4, ch)
        self.register_buffer('pos_enc', self._build_grid(self.ori_resolution))
        self.register_buffer('h_o_prev', torch.zeros(bs, self.n_objects, self.dim_h_o))
        self.register_buffer('y_e_prev', torch.zeros(bs, self.n_objects, self.dim_y_e))

        self.cte_app = cte_app
        if self.cte_app:
            self.encode_cte_rnn = nn.LSTM(feat_cte_dim, feat_cte_dim, num_layers=1, batch_first=True)

    def reset_states(self):
        for state in self.states.values():
            state.fill_(0)

    def load_states(self, *args):
        states = [self.states[arg].clone() for arg in args]
        return states if len(states) > 1 else states[0]

    def save_states(self, **kwargs):
        for kw, arg in kwargs.items():
            self.states[kw].copy_(arg.data[:, -1])
            if kw is 'h_o_prev':
                self.h_o_prev.copy_(arg.data[:, -1])
            if kw is 'y_e_prev':
                self.y_e_prev.copy_(arg.data[:, -1])


    def _build_grid(self, resolution):
        ranges = [np.linspace(0., 1., num=res) for res in resolution]
        grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
        grid = np.reshape(grid, [2, resolution[0], resolution[1]])
        axis = [0, 1]
        grid = np.expand_dims(grid, axis=axis)
        grid = grid.astype(np.float32)
        return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=len(axis)))

    def forward(self, x, dyn_feat_input=None, block='backbone'):
        bs, T, ch, h, w = x.shape

        if block == 'dyn_track':
            # TODO: Code warmup. Either cycle consistency (from T to 0 and 0 to T+K) or multiple pass in the first T0(3?) frames.
            # Initialize states for all batch
            if self.states['h_o_prev'].shape[0] == 1:
                if 0 != self.h_o_prev.norm():
                    print("Load tracker states as parameters")
                    self.states['h_o_prev'] = self.h_o_prev#.repeat_interleave(bs, dim=0).to(x.device)
                    self.states['y_e_prev'] = self.y_e_prev#.repeat_interleave(bs, dim=0).to(x.device)
                else:
                    self.states['h_o_prev'] = self.states['h_o_prev'].repeat_interleave(bs, dim=0).to(x.device)
                    self.states['y_e_prev'] = self.states['y_e_prev'].repeat_interleave(bs, dim=0).to(x.device)
                    # self.h_o_prev = self.h_o_prev.repeat_interleave(bs, dim=0).to(x.device)
                    # self.y_e_prev = self.y_e_prev.repeat_interleave(bs, dim=0).to(x.device)

            x = torch.cat([x, self.pos_enc.repeat(bs, T, 1, 1, 1)], dim=2)
            x = self.cnn_attn(x.reshape(-1, *x.shape[2:])) # TODO: Encode with lateral encoder ? #convolutional encoder
            x = x.reshape(bs, T, -1, x.shape[-1]*x.shape[-2]).permute(0, 1, 3, 2) # bs, T, hw, c_cnn
            h_o_prev, y_e_prev = self.load_states('h_o_prev', 'y_e_prev')
            h_o_seq, y_e_seq, y_l_seq, y_p_seq, Y_s_seq, Y_a_seq = self.tracker(h_o_prev, y_e_prev, x)
            # output: rnn latent vector, y_e doesnt matter, y_l layers, y_p pose, shape, app
            self.save_states(h_o_prev=h_o_seq, y_e_prev=y_e_seq)
            # Y_s_seq is like a mask, it should be of size h*w. But not sure what that is. Y_a_seq is the appearance. Use later
            pose = y_p_seq.transpose(2, 1) # bs, n_obj, T, feat_dyn_dim
            shape = Y_s_seq.transpose(2, 1)#.reshape(bs, self.n_objects, T, 1, self.cte_resolution[0], self.cte_resolution[1]) # bs, n_obj, T, 1, w, h
            app = Y_a_seq.transpose(2, 1) # bs, n_obj, T, feat_cte_dim
            confi = y_e_seq.transpose(2, 1)

            if self.cte_app: # if i decide that the appearance will be constant all the time
                bs, n_obj, T, _ = app.shape
                flat_app = app.flatten(start_dim=0, end_dim=1)
                flat_cte_app, h = self.encode_cte_rnn(flat_app) # TODO: Add fc
                app = flat_cte_app[:, -1:].repeat(1, T, 1).reshape(*app.shape)#.reshape(bs, n_obj, 1, -1)


        return pose, shape, app, confi