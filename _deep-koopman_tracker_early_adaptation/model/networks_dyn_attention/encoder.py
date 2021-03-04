import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .dynamics_attention import DynamicsAttention
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

        activation = nn.ReLU(True)
        # activation = nn.ELU(True)

        self.n_objects = n_objects
        self.feat_cte_dim = feat_cte_dim
        self.feat_dyn_dim = feat_dyn_dim

        ch = 32
        self.resolution = (16, 16)

        #TODO: substitute for a good feature extractor. Check space.
        # TODO: Group norm has to be implemented with 3d convolutions, with kernel size 1 in time. Or batchnorm.
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, ch, 7, 1, 3),  # B,  ch, 64, 64 # TODO: Bigger kernel and stride?
            activation,
            nn.Conv2d(ch, ch, 5, 2, 2),  # B,  ch, 32, 32
            activation,
            nn.Conv2d(ch, ch, 5, 2, 2),  # B,  ch, 16, 16
            activation)

        self.cnn_dyn = nn.Sequential(
            nn.Conv2d(ch, ch, 5, 2, 2),  # B,  ch,  8, 8
            activation,
            nn.Conv2d(ch, ch, 5, 2, 2),  # B, ch,  4, 4
            activation,
            nn.Conv2d(ch, ch, 5, 2, 2),  # B, ch,  2, 2
            activation,
            nn.Conv2d(ch, ch, 5, 2, 2),  # B, ch,  1,  1
            activation
        )

        # dyn_output_dim = self.n_objects * feat_dyn_dim
        self.encode_dyn_rnn = nn.LSTM(ch + feat_dyn_dim, feat_dyn_dim, num_layers=2, batch_first=True)
        # self.encode_cte_rnn = nn.LSTM(feat_cte_dim, feat_cte_dim, num_layers=1, batch_first=True)


        self.cnn_cte = nn.Sequential(
            nn.Conv2d(ch, ch, 5, 2, 2),  # B,  64,  8,  8
            activation,
            nn.Conv2d(ch, ch, 5, 2, 2),  # B,  64,  4,  4
            activation,
            nn.Conv2d(ch, ch, 5, 2, 2),  # B, 256,  1,  1
            activation,
            nn.Conv2d(ch, feat_cte_dim, 5, 2, 2),  # B, 256,  1,  1
            activation
        )

        self.linear_pos = nn.Linear(4, ch)
        self.register_buffer('grid_enc', self._build_grid(self.resolution))
        self.encoder_pos = self._soft_position_embed

        self.norm = nn.LayerNorm(ch)

        self.mlp = nn.Sequential(nn.Linear(ch, ch),
                                 activation,
                                 nn.Linear(ch, ch))

        self.attn = DynamicsAttention(self.n_objects, ch, ch, eps = 1e-8, hidden_dim = 128)
        # self.slot_attention = SlotAttention(n_objects, feat_dim)

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

            x = x.reshape(-1, *x.shape[2:])
            x = self.cnn(x)
            x = x.reshape(bs, T, *x.shape[1:])

        if block == 'dyn':
            x = x.reshape(-1, *x.shape[2:])
            x = spatial_flatten_for_sa(x)
            x = self.encoder_pos(x)
            # x = self.mlp(x) #self.norm(x)
            x = x.reshape(bs * T, *x.shape[1:]).permute(0,2,1)
            x = x.reshape(bs, T, x.shape[-2], int(np.sqrt(x.shape[-1])), int(np.sqrt(x.shape[-1])))

            x = self.cnn_dyn(x.reshape(-1, *x.shape[2:]))
            # Note: Check size and RNN ( object aware? )
            x = x.reshape(bs, T, x.shape[1])

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
            x = x.reshape(bs * T, *x.shape[2:])
            # x, attn = self.attn(x, dyn_feat_input)
            x = self.cnn_cte(x)

            # If mean
            x = x.reshape(bs, self.n_objects, T, self.feat_cte_dim)

            # If similar to DDPAE
            # x, hidden = self.encode_cte_rnn(x.reshape(bs * self.n_objects, T, self.feat_cte_dim))
            # x = x[:, -1:].reshape(bs, self.n_objects, 1, self.feat_cte_dim)

        return x

def spatial_flatten_for_sa(x):
  x = x.permute(0,2,3,1)
  return x.reshape(-1, x.shape[1] * x.shape[2], x.shape[-1])