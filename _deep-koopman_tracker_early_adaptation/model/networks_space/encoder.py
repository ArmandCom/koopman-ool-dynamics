import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .dynamics_attention import DynamicsAttention
from .coord_networks import CoordEncoder
# from model.networks.kornia import warp_affine
from utils.util import NumericalRelaxedBernoulli
from model.networks_space.s3d import Simple_S3D
from model.networks_space.tracking import TrackerArray
import numpy as np
from functools import reduce

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class ImageEncoder(nn.Module):

    def __init__(self, in_channels, feat_cte_dim, feat_dyn_dim, resolution, n_objects, ngf, n_layers):
        super(ImageEncoder, self).__init__()
        # TODO: Check lateral encoder of SPAIR.
        # TODO: Check prior on number of objects of SPAIR.
        #         # Adding relative noise to object presence, possibly to prevent trivial mapping?
        #         eps = 10e-10self.n_objects = n_objects
        #         u = Uniform(0, 1)self.feat_cte_dim = feat_cte_dim
        #         u = u.rsample(obj_log_odds.size()).to(self.device)self.feat_dyn_dim = feat_dyn_dim
        #         noise = torch.log(u + eps) - torch.log(1.0 - u + eps)self.resolution = (32, 32)
        #         obj_pre_sigmoid = (obj_log_odds + noise) / 1.0

        # TODO: Group norm has to be implemented with 3d convolutions, with kernel size 1 in time. Or batchnorm.
        self.feat_dyn_dim = feat_dyn_dim
        self.feat_cte_dim = feat_cte_dim
        # self.n_objects = reduce((lambda x, y: x * y), resolution)
        self.n_objects = n_objects
        self.tau = 1
        self.hard = True
        self.ori_resolution = (128, 128)
        self.cte_resolution = (32, 32)

        last_hidden_dim = 96
        # self.encode_dyn_rnn = nn.LSTM(last_hidden_dim + self.feat_dyn_dim, self.feat_dyn_dim, num_layers=1, batch_first=True)
        # self.encode_cte_rnn = nn.LSTM(feat_cte_dim, feat_cte_dim, num_layers=1, batch_first=True)

        # # CTE
        # self.cnn_cte_bb = nn.Sequential(
        #     nn.Conv2d(in_channels, 16, 4, 2, 1), # 32, 32
        #     nn.CELU(),
        #     nn.BatchNorm2d(16),
        #     nn.Conv2d(16, 32, 4, 2, 1), # 16, 16
        #     nn.CELU(),
        #     nn.BatchNorm2d(32),
        #     nn.Conv2d(32, 64, 4, 2, 1),
        #     nn.CELU(),
        #     nn.BatchNorm2d(64),
        # )
        #
        # self.cnn_cte = nn.Sequential(
        #     nn.Conv2d(64, 64, 4, 2, 1),
        #     nn.CELU(),
        #     nn.BatchNorm2d(64),
        #     nn.Conv2d(64, 128, 4, 2, 1),
        #     nn.CELU(),
        #     nn.BatchNorm2d(128),
        #     # nn.Conv2d(128, 128, 4, 2, 1),
        #     # nn.CELU(),
        #     # nn.BatchNorm2d(128),
        #     nn.Conv2d(128, self.feat_cte_dim, 1),
        #     nn.CELU(),
        #     nn.BatchNorm2d(self.feat_cte_dim),
        #     nn.Conv2d(self.feat_cte_dim, self.feat_cte_dim, 1),
        # )


        # DYN
        # TODO: Simplify output a great deal. Maxpool and similar is permitted (AND DESIRED) as we dont need or want to keep all info.
        #  - I3D reduced version S3D
        # self.cnn_dyn_s3d = Simple_S3D(chan_out=self.feat_dyn_dim, n_obj=self.n_objects, dropout_keep_prob = 0.8, input_channel = 1, spatial_squeeze=True)

        # self.cnn_dyn = nn.Sequential(
        #     nn.Conv2d(in_channels, 16, 4, 2, 1),
        #     nn.CELU(),
        #     nn.BatchNorm2d(16),
        #     nn.Conv2d(16, 32, 4, 2, 1),
        #     nn.CELU(),
        #     nn.BatchNorm2d(32),
        #     nn.Conv2d(32, 64, 4, 2, 1),
        #     nn.CELU(),
        #     nn.BatchNorm2d(64),
        #     nn.Conv2d(64, 64, 4, 2, 1),
        #     nn.CELU(),
        #     nn.BatchNorm2d(64),
        #     nn.Conv2d(64, last_hidden_dim, 4, 2, 1),
        #     nn.CELU(),
        #     nn.BatchNorm2d(last_hidden_dim),
        #     nn.Conv2d(last_hidden_dim, last_hidden_dim, 4, 2, 1),
        #     nn.CELU(),
        #     nn.BatchNorm2d(last_hidden_dim),
        #     nn.Conv2d(last_hidden_dim, last_hidden_dim, 1),
        #     # nn.CELU(),
        #     # nn.BatchNorm2d(self.feat_cte_dim)
        # )

        # TODO: Positional encoding
        self.cnn_attn = nn.Sequential(
            nn.Conv2d(in_channels + 4, 16, 4, 2, 1), # 32, 32
            nn.CELU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.CELU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 4, 2, 1), # 16, 16
            nn.CELU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 4, 2, 1), # 16, 16
            nn.CELU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 4, 2, 1), # 8, 8
            nn.CELU(),
            nn.BatchNorm2d(64),
            # nn.Conv2d(64, last_hidden_dim, 4, 2, 1), # 4, 4
            # nn.CELU(),
            # nn.BatchNorm2d(last_hidden_dim),
            # nn.Conv2d(last_hidden_dim, last_hidden_dim, 4, 1, 1),
            # nn.CELU(),
            # nn.BatchNorm2d(last_hidden_dim),
            nn.Conv2d(64, last_hidden_dim, 3, 1, 1),
            nn.CELU(),
            nn.BatchNorm2d(last_hidden_dim),
            nn.Conv2d(last_hidden_dim, last_hidden_dim, 1),
        )

        '''Attention params'''
        # self.z_pres_net = nn.Conv2d(last_hidden_dim, 1, 1)
        # self.attn = DynamicsAttention(in_dim=last_hidden_dim, out_dim=feat_dyn_dim, padding=1, resolution=resolution)

        '''Tracker params'''
        dims = {'confidence': 1,
                'layer': 1,
                'pose': feat_dyn_dim,
                'shape': 1, #self.cte_resolution[0]*self.cte_resolution[1],
                'appearance':feat_cte_dim,
                'tracker_state':last_hidden_dim * 3,
                'features':last_hidden_dim}

        self.dim_h_o = dims['tracker_state']
        self.dim_y_e = dims['confidence']
        self.tracker = TrackerArray(self.n_objects, dims)
        # States
        self.states = {}
        self.states['h_o_prev'] = torch.Tensor(1, self.n_objects, self.dim_h_o)
        self.states['y_e_prev'] = torch.Tensor(1, self.n_objects, self.dim_y_e)
        self.reset_states()

        # self.linear_pos = nn.Linear(4, ch)
        self.register_buffer('pos_enc', self._build_grid(self.ori_resolution))

        '''Temporal aggregation'''
        # self.temp_aggr = TempNet(2, feat_cte_dim, feat_cte_dim * 2, feat_cte_dim, do_prob=0.2)

        self.gamma = nn.Parameter(torch.zeros(1))

    def reset_states(self):
        for state in self.states.values():
            state.fill_(0)

    def load_states(self, *args):
        states = [self.states[arg].clone() for arg in args]
        return states if len(states) > 1 else states[0]

    def save_states(self, **kwargs):
        for kw, arg in kwargs.items():
            self.states[kw].copy_(arg.data[:, -1])

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

        if block == 'backbone':
            pass
            # x = x.reshape(-1, *x.shape[2:])
            # x = self.cnn(x)
            # x = x.reshape(bs, T, *x.shape[1:])

        if block == 'dyn_cnn':
            x = self.cnn_dyn_s3d(x.permute(0, 2, 1, 3, 4)) # bs, ch, T, h, w
            x = x.reshape(bs, self.n_objects, self.feat_dyn_dim, T).transpose(-1, -2) # bs, n_obj, T, feat_dyn_dim

        if block == 'dyn_attn':
            # TODO: Check neurips 2020 for similarity. Similarity (aka. qk without qk) at higher resolution and parallel to main dyn feat extraction.
            x = self.cnn_attn(x.reshape(-1, *x.shape[2:])) # TODO: Encode with lateral encoder ?
            # x = x.reshape(bs, T, *x.shape[1:])
            # TODO: Penalize bounding box size.

            # z_pres_logits = 8.8 * torch.tanh(self.z_pres_net(x)).reshape(bs, T, 1, *x.shape[2:])
            #
            # # (B, 1, G, G) - > (B, G*G, 1)
            # B, T, D, G, G = z_pres_logits.size()
            # z_pres_logits = z_pres_logits.permute(0, 3, 4, 1, 2).view(B, G * G, T, D)
            # # z_pres_prob = torch.sigmoid(z_pres_logits)
            #
            # z_pres_post = NumericalRelaxedBernoulli(logits=z_pres_logits, temperature=self.tau)
            # # Unbounded
            # z_pres_y = z_pres_post.rsample()
            # # in (0, 1)
            # z_pres = torch.sigmoid(z_pres_y)
            #
            # if self.hard:
            #     z_pres_soft = z_pres
            #     zeros = torch.zeros_like(z_pres_soft)
            #     ones = 1 - zeros
            #     z_pres_hard = torch.where(z_pres_soft > 0.5, ones, zeros)
            #     z_pres = z_pres_hard - z_pres_soft.detach() + z_pres_soft

            x, attn = self.attn(x.reshape(bs, T, *x.shape[1:])) # bs, n_obj(hw), T, feat_dyn_dim

            z_pres, z_pres_post = None, None

            return x, attn, z_pres, z_pres_post

        if block == 'dyn_track':

            # Initialize states for all batch
            if self.states['h_o_prev'].shape[0] == 1:
                self.states['h_o_prev'] = self.states['h_o_prev'].repeat_interleave(bs, dim=0).to(x.device)
                self.states['y_e_prev'] = self.states['y_e_prev'].repeat_interleave(bs, dim=0).to(x.device)

            x = torch.cat([x, self.pos_enc.repeat(bs, T, 1, 1, 1)], dim=2)
            x = self.cnn_attn(x.reshape(-1, *x.shape[2:])) # TODO: Encode with lateral encoder ?
            x = x.reshape(bs, T, -1, x.shape[-1]*x.shape[-2]).permute(0, 1, 3, 2) # bs, T, hw, c_cnn
            h_o_prev, y_e_prev = self.load_states('h_o_prev', 'y_e_prev')
            h_o_seq, y_e_seq, y_l_seq, y_p_seq, Y_s_seq, Y_a_seq = self.tracker(h_o_prev, y_e_prev, x)
            self.save_states(h_o_prev=h_o_seq, y_e_prev=y_e_seq)
            # Y_s_seq is like a mask, it should be of size h*w. But not sure what that is. Y_a_seq is the appearance. Use later
            pose = y_p_seq.transpose(2, 1) # bs, n_obj, T, feat_dyn_dim
            shape = Y_s_seq.transpose(2, 1)#.reshape(bs, self.n_objects, T, 1, self.cte_resolution[0], self.cte_resolution[1]) # bs, n_obj, T, 1, w, h
            app = Y_a_seq.transpose(2, 1) # bs, n_obj, T, feat_cte_dim
            confi = y_e_seq.transpose(2, 1)
            return pose, shape, app, confi

        if block == 'dyn_rnn':
            x = self.cnn_dyn(x.reshape(-1, *x.shape[2:]))
            # Note: Check size and RNN ( object aware? )
            x = x.reshape(bs, T, x.shape[1])

            # TODO: Better tracking techniques
            out_objects = []
            prev_obj_hidden = [ torch.zeros(bs, 1, self.feat_dyn_dim).to(x.device) ] * T
            for i in range(self.n_objects):
                out_time = []
                hidden = None
                for t in range(T):
                    rnn_input = torch.cat([x[:, t:t+1, :], prev_obj_hidden[t]], dim=-1) # bs, 1, feat_dyn_dim + ch
                    output, hidden = self.encode_dyn_rnn(rnn_input, hidden)
                    h = hidden[0][-1]
                    prev_obj_hidden[t] = h.view(bs, 1, self.feat_dyn_dim)
                    out_time.append(output)
                out_objects.append(torch.cat(out_time, dim=1)) # bs, T, feat_dyn_dim
            x = torch.stack(out_objects, dim=1) # bs, n_obj, T, feat_dyn_dim

        if block == 'cte':

            # Option 2
            x = self.cnn_cte_bb(x.reshape(-1, *x.shape[-3:])) # TODO: add dyn_feat_input to the input channels (mid depth)
            x = self.cnn_cte(x)

            # # Different ways to combine temporal features.

            # TORRALBA CAUSALITY PAPER. Time combining CNN
            # x = self.temp_aggr(x, T)  # Make sure objects come before time. Because it reshapes according to this.
            # x = x * self.mask.reshape(x.shape[0], -1)
            # x = x.reshape(bs, self.n_objects, 1, self.feat_cte_dim)

            # Mixing variable.
            # temp = 5
            # weighting = self.weighting(x)
            # x = x.reshape(bs, self.n_objects, T, self.feat_cte_dim)
            # weighting = weighting.reshape(bs, self.n_objects, T, 1)
            # x = x * F.softmax(weighting/temp, dim=-2)

            # Normal
            # x = x.reshape(bs, self.n_objects, T, self.feat_cte_dim)

            # Cte
            # x = x.reshape(bs, self.n_objects, 1, self.feat_cte_dim)

            # If softmax()
            # x = self.attn.forward_last(x)

            # If mean
            # x = x.reshape(bs, self.n_objects, T, self.feat_cte_dim)[:, :, None].mean(3)

            # If RNN
            x, hidden = self.encode_cte_rnn(x.reshape(bs, T, self.feat_cte_dim))
            x = x[:, -1]

            # x = x[:, -1:].reshape(bs, self.n_objects, 1, self.feat_cte_dim)

        return x

class TempNet(nn.Module):
    def __init__(self, ks, nf_in, nf_hidden, nf_out, do_prob=0.2):
        super(TempNet, self).__init__()

        self.nf_in = nf_in
        self.nf_out = nf_out

        self.pool = nn.MaxPool1d(
            kernel_size=2, stride=None, padding=0,
            dilation=1, return_indices=False,
            ceil_mode=False)

        self.conv1 = nn.Conv1d(nf_in, nf_hidden, kernel_size=ks, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(nf_hidden)
        self.conv2 = nn.Conv1d(nf_hidden, nf_hidden, kernel_size=ks, stride=1, padding=0)
        self.bn2 = nn.BatchNorm1d(nf_hidden)
        self.conv3 = nn.Conv1d(nf_hidden, nf_hidden, kernel_size=ks, stride=1, padding=0)
        self.bn3 = nn.BatchNorm1d(nf_hidden)
        self.conv_predict = nn.Conv1d(nf_hidden, nf_out, kernel_size=1)
        # self.conv_attention = nn.Conv1d(nf_hidden, 1, kernel_size=1)
        self.dropout_prob = do_prob

    def forward(self, inputs, T):
        # inputs: B x T x nf_in
        inputs = inputs.reshape(-1, T, self.nf_in)
        inputs = inputs.transpose(1, 2)

        # inputs: B x nf_in x T
        x = F.celu(self.conv1(inputs))
        x = self.bn1(x)
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = self.pool(x)
        # x = F.celu(self.conv2(x))
        # x = self.bn2(x)
        # x = F.dropout(x, self.dropout_prob, training=self.training)
        # x = self.pool(x)
        x = F.celu(self.conv3(x))
        x = self.bn3(x)
        pred = self.conv_predict(x)
        # ret: B x nf_out
        ret = pred.max(dim=2)[0]
        return ret

class TempNet(nn.Module):
    def __init__(self, ks, nf_in, nf_hidden, nf_out, do_prob=0.2):
        super(TempNet, self).__init__()

        self.nf_in = nf_in
        self.nf_out = nf_out

        self.pool = nn.MaxPool1d(
            kernel_size=2, stride=None, padding=0,
            dilation=1, return_indices=False,
            ceil_mode=False)

        self.cnn_enc = nn.Sequential(
            nn.Conv3d(3 + 4, 16, [3,9,9], [1,1,1], [0,4,4]), # T, 128, 128
            nn.CELU(),
            nn.BatchNorm2d(16),
            nn.Conv3d(16, 16, [3,4,4], [1,2,2], 1), # T, 128, 128
            nn.CELU(),
            nn.BatchNorm2d(16),
            nn.Conv3d(16, 32, [3,3,3], 1, 1), # 64, 164
            nn.CELU(),
            nn.BatchNorm3d(32),
            # nn.AvgPool3d(kernel_size=3)
            # nn.Dropout2d(0.3),
            nn.Conv2d(32, 64, [3,4,4], [1,2,2], 1), # 32, 32
            nn.CELU(),
            nn.BatchNorm3d(64),
            nn.Conv2d(64, 64, 1), # 32, 32
        )
        self.cnn_dec = nn.Sequential(
            nn.Conv2d(64, 32 * 2 * 2, 1), # T, 64, 64
            nn.PixelShuffle(2),
            nn.CELU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(32, 16 * 2 * 2, 1), # T, 128, 128
            nn.PixelShuffle(2),
            nn.CELU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 3, 1), # T, 64, 64
        )
        self.conv1 = nn.Conv1d(nf_in, nf_hidden, kernel_size=ks, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(nf_hidden)
        self.conv2 = nn.Conv1d(nf_hidden, nf_hidden, kernel_size=ks, stride=1, padding=0)
        self.bn2 = nn.BatchNorm1d(nf_hidden)
        self.conv3 = nn.Conv1d(nf_hidden, nf_hidden, kernel_size=ks, stride=1, padding=0)
        self.bn3 = nn.BatchNorm1d(nf_hidden)
        self.conv_predict = nn.Conv1d(nf_hidden, nf_out, kernel_size=1)
        # self.conv_attention = nn.Conv1d(nf_hidden, 1, kernel_size=1)
        self.dropout_prob = do_prob

    def forward(self, inputs, T):
        # inputs: B x T x nf_in
        # inputs = inputs.reshape(-1, T, self.nf_in)
        # inputs = inputs.transpose(1, 2)

        x = self.cnn_enc(inputs)
        x # [BS, T//4, 32, 32]
        x = x.max(dim=1)[0] # Max pooling [BS, 1, 32, 32]
        x = self.cnn_dec(x)
        # inputs: B x nf_in x T
        x = F.celu(self.conv1(inputs))
        x = self.bn1(x)
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = self.pool(x)
        # x = F.celu(self.conv2(x))
        # x = self.bn2(x)
        # x = F.dropout(x, self.dropout_prob, training=self.training)
        # x = self.pool(x)
        x = F.celu(self.conv3(x))
        x = self.bn3(x)
        pred = self.conv_predict(x)
        # ret: B x nf_out
        ret = pred.max(dim=2)[0]
        return ret