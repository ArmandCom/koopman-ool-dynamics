import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from base import BaseModel
from model.networks import ImageEncoder, SBImageDecoder, SimpleSBImageDecoder, KoopmanOperators, AttImageEncoder
from model.networks_cswm.modules import TransitionGNN, EncoderCNNLarge, EncoderCNNMedium, EncoderCNNSmall, EncoderMLP, DecoderCNNLarge, DecoderCNNMedium, DecoderCNNSmall, DecoderMLP
import matplotlib.pyplot as plt

''' A main objective of the model is to separate the dimensions to be modeled/propagated by koopman operator
    from the constant dimensions - elements that remain constant - which should be the majority of info. 
    This way we only need to control a reduced amount of dimensions - and we can make them variable.
    The constant factors can also work as a padding for the dimensionality to remain constant. '''

class _KoopmanModel(BaseModel):
    def __init__(self, in_channels, feat_dim, nf_particle, nf_effect, g_dim,
                 I_factor=10, psteps=1, n_timesteps=1, ngf=8, image_size=[64, 64]):
        super().__init__()
        out_channels = 1
        n_layers = int(np.log2(image_size[0])) - 1
        num_objects = 1

        # Set state dim with config, depending on how many time-steps we want to take into account
        self.image_size = image_size
        self.n_timesteps = n_timesteps
        self.state_dim = feat_dim
        self.I_factor = I_factor
        self.psteps = psteps
        self.g_dim = g_dim

        self.image_encoder = EncoderCNNLarge(
            input_dim =in_channels,
            hidden_dim =  feat_dim // 8,
            output_dim = feat_dim,
            num_objects = num_objects)

        self.obj_encoder = EncoderMLP(
            input_dim=np.prod(image_size)//(8**2) * feat_dim,
            hidden_dim=feat_dim // 2,
            output_dim=feat_dim,
            num_objects=num_objects)

        self.state_encoder = EncoderMLP(
            input_dim=feat_dim * self.n_timesteps,
            hidden_dim=feat_dim,
            output_dim=g_dim,
            num_objects=num_objects)

        self.state_decoder = EncoderMLP(
            input_dim=g_dim,
            hidden_dim=feat_dim,
            output_dim=feat_dim,
            num_objects=num_objects)

        # self.transition_model = TransitionGNN(
        #     input_dim=embedding_dim,
        #     hidden_dim=hidden_dim,
        #     action_dim=action_dim,
        #     num_objects=num_objects,
        #     ignore_action=ignore_action,
        #     copy_action=copy_action)

        self.image_decoder = SBImageDecoder(feat_dim, out_channels, ngf, n_layers, image_size)
        self.koopman = KoopmanOperators(self.state_dim, nf_particle, nf_effect, g_dim, n_timesteps)

    def _add_positional_encoding(self, x):

        x = torch.cat((self.x_grid_enc.expand(*x.shape[:2], -1, -1, -1),
                           self.y_grid_enc.expand(*x.shape[:2], -1, -1, -1),
                           x), dim=-3)

        return x

    def _get_full_state(self, x, T):

        new_T = T - self.n_timesteps + 1
        x = x.reshape(-1, T, *x.shape[1:])
        new_x = []
        for t in range(new_T):
            new_x.append(torch.cat([x[:, t+idx] for idx in range(self.n_timesteps)], dim=-1))

        new_x = torch.stack(new_x, dim=1)
        return new_x.reshape(-1, *new_x.shape[2:]), new_T

    def forward(self, input):
        bs, T, ch, h, w = input.shape

        cnnf = self.image_encoder(input.reshape(-1, ch, h, w))
        f = self.obj_encoder(cnnf)

        # Note: test image encoder with attention
        # f_mu, f_logvar = torch.chunk(self.att_image_encoder(input.reshape(-1, ch, h, w)), 2, dim=-1)
        # f = _sample(f_mu, f_logvar)

        # input = self._add_positional_encoding(input)

        f, T = self._get_full_state(f, T)
        # f = f.view(torch.Size([bs, T]) + f.size()[1:])
        # g, G_for_pred = f, f

        g = self.koopman.to_g(f, self.psteps)
        g = g.view(torch.Size([bs, T]) + g.size()[1:])
 
        G_tilde = g[:, :-1, None] # new axis corresponding to N number of objects
        H_tilde = g[:, 1:, None]

        A, fit_err = self.koopman.system_identify(G=G_tilde, H=H_tilde, I_factor=self.I_factor) # TODO: maybe

        ''' rollout: BT x D '''
        G_for_pred = self.koopman.simulate(T=T-1, g=g[:, 0], A=A) # Rollout from initial observation to the last

        s_for_rec = self.koopman.to_s(gcodes=_get_flat(g), pstep=self.psteps)
        s_for_pred = self.koopman.to_s(gcodes=_get_flat(G_for_pred), pstep=self.psteps)

        out_rec =  self.image_decoder(s_for_rec)
        out_pred = self.image_decoder(s_for_pred)

        o_touple = (out_rec, out_pred, g.reshape(-1, g.size(-1)), f_mu.reshape(-1, f_mu.size(-1)), f_logvar.reshape(-1, f_logvar.size(-1)))
        o = [item.reshape(torch.Size([bs, -1]) + item.size()[1:]) for item in o_touple]

        return o

class KoopmanModel(BaseModel):
    def __init__(self, in_channels, feat_dim, nf_particle, nf_effect, g_dim,
                 I_factor=10, psteps=1, n_timesteps=1, ngf=8, image_size=[64, 64]):
        super().__init__()
        out_channels = 1
        n_layers = int(np.log2(image_size[0])) - 1

        # Positional encoding buffers
        x = torch.linspace(-1, 1, image_size[0])
        y = torch.linspace(-1, 1, image_size[1])
        x_grid, y_grid = torch.meshgrid(x, y)
        # Add as constant, with extra dims for N and C
        self.register_buffer('x_grid_enc', x_grid.view((1, 1, 1) + x_grid.shape))
        self.register_buffer('y_grid_enc', y_grid.view((1, 1, 1) + y_grid.shape))

        # Set state dim with config, depending on how many time-steps we want to take into account
        self.image_size = image_size
        self.n_timesteps = n_timesteps
        self.state_dim = feat_dim
        self.I_factor = I_factor
        self.psteps = psteps
        self.g_dim = g_dim

        self.att_image_encoder = AttImageEncoder(out_channels, feat_dim, ngf, n_layers)
        self.image_encoder = ImageEncoder(in_channels, self.state_dim, ngf, n_layers)  # feat_dim * 2 if sample here
        self.image_decoder = SimpleSBImageDecoder(feat_dim, out_channels, ngf, n_layers, image_size)
        self.koopman = KoopmanOperators(self.state_dim, nf_particle, nf_effect, g_dim, n_timesteps)

    def _add_positional_encoding(self, x):

        x = torch.cat((self.x_grid_enc.expand(*x.shape[:2], -1, -1, -1),
                       self.y_grid_enc.expand(*x.shape[:2], -1, -1, -1),
                       x), dim=-3)

        return x

    def _get_full_state(self, x, T):

        new_T = T - self.n_timesteps + 1
        x = x.reshape(-1, T, *x.shape[1:])
        new_x = []
        for t in range(new_T):
            new_x.append(torch.cat([x[:, t + idx] for idx in range(self.n_timesteps)], dim=-1))

        new_x = torch.stack(new_x, dim=1)
        return new_x.reshape(-1, *new_x.shape[2:]), new_T

    def forward(self, input):
        bs, T, ch, h, w = input.shape

        # Note: test image encoder with attention
        # f_mu, f_logvar = torch.chunk(self.att_image_encoder(input.reshape(-1, ch, h, w)), 2, dim=-1)
        # f = _sample(f_mu, f_logvar)

        # # Show images
        # plt.imshow(input[0, 0, :].reshape(16, 16).cpu().detach().numpy())
        # plt.savefig('test_attention.png')

        # Note: Koopman applied to mu and logvar
        f = self.att_image_encoder(input.reshape(-1, ch, h, w))

        # input = self._add_positional_encoding(input)
        # f = self.image_encoder(input.reshape(-1, ch+2, h, w)) #Note: deterministic AE
        # f_mu, f_logvar = f, f
        # f_mu, f_logvar = torch.chunk(self.image_encoder(input.reshape(-1, ch+2, h, w)), 2, dim=-1)
        # f = _sample(f_mu, f_logvar)
        # f = f_mu

        # # Concatenates the features from current time and previous to create full state
        # f, T = self._get_full_state(f, T)
        #  f = f.view(torch.Size([bs, T]) + f.size()[1:])
        # g, G_for_pred = f, f

        g = self.koopman.to_g(f, self.psteps)
        g = g.view(torch.Size([bs, T]) + g.size()[1:])

        # # Note: Ignore - Split representation into constant and dynamic by hardcoding
        # g, g_cte = torch.chunk(g, 2, dim=-1) # Option 1
        # g, g_cte = g[..., :2], g[..., 2:] # Option 2

        G_tilde = g[:, :-1, None]  # new axis corresponding to N number of objects
        H_tilde = g[:, 1:, None]

        # Find Koopman operator from current data (matrix a)
        A, fit_err = self.koopman.system_identify(G=G_tilde, H=H_tilde, I_factor=self.I_factor)

        # Rollout. From observation in time 0, predict with Koopman operator
        G_for_pred = self.koopman.simulate(T=T - 1, g=g[:, 0], A=A)  # Rollout from initial observation to the last

        # Note: Ignore - If we have split the representation: Merge constant and dynamic components
        if g.size(-1) < self.g_dim:
            g = torch.cat([g, g_cte[:, 0:1].repeat(1, T, 1)], dim=-1)
            G_for_pred = torch.cat([G_for_pred, g_cte[:, 0:1].repeat(1, T - 1, 1)], dim=-1)


        g_mu, g_logvar = torch.chunk(g, 2, dim=-1)
        g_mu_pred, g_logvar_pred = torch.chunk(G_for_pred, 2, dim=-1)
        g = _sample(g_mu, g_logvar)
        G_for_pred = _sample(g_mu_pred, g_logvar_pred)
        f_mu, f_logvar = g_mu, g_logvar

        # Option 1: use the koopman object decoder
        # s_for_rec = self.koopman.to_s(gcodes=_get_flat(g),
        #                               pstep=self.psteps)
        # s_for_pred = self.koopman.to_s(gcodes=_get_flat(torch.cat([g[:, :1],G_for_pred], dim=1)),
        #                                pstep=self.psteps)
        # Option 2: we don't use the koopman object decoder
        s_for_rec = _get_flat(g)
        s_for_pred = _get_flat(G_for_pred)

        # Convolutional decoder. Normally Spatial Broadcasting decoder
        out_rec = self.image_decoder(s_for_rec)
        out_pred = self.image_decoder(s_for_pred)

        o_touple = (out_rec, out_pred, f_mu.reshape(-1, f_mu.size(-1)), f_mu.reshape(-1, f_mu.size(-1)),
                    f_logvar.reshape(-1, f_logvar.size(-1)))
        o = [item.reshape(torch.Size([bs, -1]) + item.size()[1:]) for item in o_touple]
        # Returns reconstruction, prediction, g_representation, mu and sigma from which we sample to compute KL divergence
        return o

def _get_flat(x, keep_dim=False):
    if keep_dim:
        return x.reshape(torch.Size([1, x.size(0) * x.size(1)]) + x.size()[2:])
    return x.reshape(torch.Size([x.size(0) * x.size(1)]) + x.size()[2:])

# If we want it to be variational:
def _sample(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std