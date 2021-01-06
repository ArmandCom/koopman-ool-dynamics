import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .dynamics_attention import DynamicsAttention
from .coord_networks import CoordEncoder
from .s3d import Simple_S3DG
import numpy as np

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class ImageEncoder(nn.Module):

    def __init__(self, in_channels, feat_cte_dim, feat_dyn_dim, n_objects, ngf, n_layers):
        super(ImageEncoder, self).__init__()

        self.n_objects = n_objects
        self.feat_cte_dim = feat_cte_dim
        self.feat_dyn_dim = feat_dyn_dim
        self.resolution = (8, 8)

        # TODO: Group norm has to be implemented with 3d convolutions, with kernel size 1 in time. Or batchnorm.

        last_hidden_dim = 128
        self.encode_dyn_rnn = nn.LSTM(last_hidden_dim + self.feat_dyn_dim, self.feat_dyn_dim, num_layers=1, batch_first=True)
        # self.encode_cte_rnn = nn.LSTM(feat_cte_dim, feat_cte_dim, num_layers=1, batch_first=True)

        # CTE
        self.cnn_cte_bb = nn.Sequential(
            nn.Conv2d(in_channels, 16, 4, 2, 1), # 32, 32
            nn.CELU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 4, 2, 1), # 16, 16
            nn.CELU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.CELU(),
            nn.BatchNorm2d(64),
        )

        self.cnn_cte = nn.Sequential(
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.CELU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.CELU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 4, 2, 1),
            nn.CELU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, self.feat_cte_dim, 1),
            nn.CELU(),
            nn.BatchNorm2d(self.feat_cte_dim),
            nn.Conv2d(self.feat_cte_dim, self.feat_cte_dim, 1),
        )


        # DYN
        # TODO: Simplify output a great deal. Maxpool and similar is permitted (AND DESIRED) as we dont need or want to keep all info.
        #  - I3D reduced version S3D
        self.cnn_dyn = Simple_S3DG(chan_out=self.feat_dyn_dim, n_obj=self.n_objects, dropout_keep_prob = 0.8, input_channel = 1, spatial_squeeze=True)

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

        # self.last_layer(last_hidden_dim, last_hidden_dim)

        # self.linear_pos = nn.Linear(4, ch)
        # self.register_buffer('grid_enc', self._build_grid(self.resolution))
        # self.encoder_pos = self._soft_position_embed
        #
        # self.norm = nn.LayerNorm(ch)
        #
        # self.mlp = nn.Sequential(nn.Linear(ch, ch),
        #                          activation,
        #                          nn.Linear(ch, ch))

        # self.warping = CoordEncoder(self.feat_cte_dim, self.feat_dyn_dim//2, feat_map_size=self.resolution)
        self.warping = CoordEncoder(1, self.feat_dyn_dim//2, feat_map_size=(64, 64))

        self.temp_aggr = TempNet(2, feat_cte_dim, feat_cte_dim * 2, feat_cte_dim, do_prob=0.2)

        # self.attn = DynamicsAttention(self.n_objects, in_dim=32, dim=32, last_dim=self.feat_cte_dim, dyn_dim=self.feat_dyn_dim//2, feat_map_size=self.resolution, eps = 1e-8, hidden_dim = 64)

        self.gamma = nn.Parameter(torch.zeros(1))

    def _soft_position_embed(self, input):
        return input + self.gamma * self.linear_pos(self.grid_enc)

    def _build_grid(self, resolution):
        ranges = [np.linspace(0., 1., num=res) for res in resolution]
        grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
        grid = np.stack(grid, axis=-1)
        grid = np.reshape(grid, [resolution[0], resolution[1], -1])
        grid = np.expand_dims(grid, axis=0)
        grid = grid.astype(np.float32)
        return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1)).reshape(grid.shape[0], -1, 4)

    def forward(self, x, dyn_feat_input=None, block='backbone'):
        bs, T, ch, h, w = x.shape

        if block == 'backbone':
            pass
            # x = x.reshape(-1, *x.shape[2:])
            # x = self.cnn(x)
            # x = x.reshape(bs, T, *x.shape[1:])

        if block == 'dyn_cnn':
            x = self.cnn_dyn(x.permute(0, 2, 1, 3, 4)) # bs, ch, T, h, w
            x = x.reshape(bs, self.n_objects, self.feat_dyn_dim, T).transpose(-1, -2) # bs, n_obj, T, feat_dyn_dim

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
            assert dyn_feat_input is not None

            # TODO: Separar dinamica euleriana de lagrangiana.
            # TODO: concat dyn_feat to normal features to get a mask (self.attn) and other kind of dynamics.
            # Option 1
            # x = self.cnn_cte_bb(x.reshape(-1, *x.shape[2:]))
            # x = x.reshape(bs, T, *x.shape[1:])
            # # x, attn = self.attn(x, dyn_feat_input)
            # x, MF = self.warping(x, dyn_feat_input)
            # x = self.cnn_cte(x)

            # Option 2
            x, MF = self.warping(x, dyn_feat_input)
            x = self.cnn_cte_bb(x) # TODO: add dyn_feat_input to the input channels (mid depth)
            x = self.cnn_cte(x)

            # # Different ways to combine temporal features.

            # TORRALBA CAUSALITY PAPER. Time combining CNN
            x = self.temp_aggr(x, T)
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
            # x, hidden = self.encode_cte_rnn(x.reshape(bs * self.n_objects, T, self.feat_cte_dim))
            # x = x[:, -1:].reshape(bs, self.n_objects, 1, self.feat_cte_dim)

            return x, MF
        return x

def spatial_flatten_for_sa(x):
  x = x.permute(0,2,3,1)
  return x.reshape(-1, x.shape[1] * x.shape[2], x.shape[-1])


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