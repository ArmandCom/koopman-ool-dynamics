import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from base import BaseModel
from model.networks import ImageEncoder, SBImageDecoder, KoopmanOperators

''' A main objective of the model is to separate the dimensions to be modeled/propagated by koopman operator
    from the constant dimensions - elements that remain constant - which should be the majority of info. 
    This way we only need to control a reduced amount of dimensions - and we can make them variable.
    The constant factors can also work as a padding for the dimensionality to remain constant. '''

class KoopmanModel(BaseModel):
    def __init__(self, in_channels, feat_dim, nf_particle, nf_effect, g_dim, ngf=8, image_size=[64, 64]):
        super().__init__()
        # Note: I_factor, psteps as argument
        out_channels = 1
        n_layers = int(np.log2(image_size[0])) - 1
        self.state_dim = feat_dim
        self.I_factor = 10
        self.psteps = 2

        self.image_encoder = ImageEncoder(in_channels, feat_dim, ngf, n_layers)
        self.image_decoder = SBImageDecoder(feat_dim, out_channels, ngf, n_layers, image_size)
        self.koopman = KoopmanOperators(self.state_dim, nf_particle, nf_effect, g_dim)

    def forward(self, input):
        # Note: Two ways: either diagonal or block diagonal with blocks 2x2 (obs + 1st derivative)
        bs, T, ch, h, w = input.shape
        f = self.image_encoder(input.reshape(-1, ch, h, w))
        g = self.koopman.to_g(f, self.psteps)
        g = g.view(torch.Size([bs, T]) + g.size()[1:])

        G_tilde = g[:, :-1, None] # new axis corresponding to N number of objects
        H_tilde = g[:, 1:, None]

        # Note: We fit with a minibatch. One A for all samples in the minibatch. This matrix would be used with different data.
        # G_tilde = _get_flat(G_tilde, keep_dim=True)
        # H_tilde = _get_flat(H_tilde, keep_dim=True)

        A, fit_err = self.koopman.system_identify(G=G_tilde, H=H_tilde, I_factor=self.I_factor)
        # self.koopman.A = self.koopman.A.repeat(bs, 1, 1)

        ''' rollout: BT x D '''
        G_for_pred = self.koopman.simulate(T=T-1, g=g[:, 0], A=A) # Rollout from initial observation to the last
        s_for_rec = self.koopman.to_s(gcodes=_get_flat(g), pstep=self.psteps)
        s_for_pred = self.koopman.to_s(gcodes=_get_flat(G_for_pred), pstep=self.psteps)

        out_rec =  self.image_decoder(s_for_rec)
        out_pred = self.image_decoder(s_for_pred)

        o_list = (out_rec, out_pred, g)
        o = [item.reshape(torch.Size([bs, -1]) + item.size()[1:]) for item in o_list]

        return o

def _get_flat(x, keep_dim=False):
    if keep_dim:
        return x.reshape(torch.Size([1, x.size(0) * x.size(1)]) + x.size()[2:])
    return x.reshape(torch.Size([x.size(0) * x.size(1)]) + x.size()[2:])