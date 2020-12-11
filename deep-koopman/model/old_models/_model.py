import torch
import torch.nn as nn
# import torch.nn.functional as F
import numpy as np
from base import BaseModel
# from model.networks import ImageEncoder, SBImageDecoder, SimpleSBImageDecoder, \
#     KoopmanOperators, AttImageEncoder, ImageDecoder, \
#     deconvSpatialDecoder, linearSpatialDecoder, LinearEncoder, ObjectAttention
from model.networks import KoopmanOperators
from model.networks_slot_attention import ImageEncoder, ImageDecoder
# from model.networks_cswm.modules import TransitionGNN, EncoderCNNLarge, EncoderCNNMedium, EncoderCNNSmall, EncoderMLP, DecoderCNNLarge, DecoderCNNMedium, DecoderCNNSmall, DecoderMLP
from torch.autograd import Variable
import matplotlib.pyplot as plt

''' A main objective of the model is to separate the dimensions to be modeled/propagated by koopman operator
    from the constant dimensions - elements that remain constant - which should be the majority of info. 
    This way we only need to control a reduced amount of dimensions - and we can make them variable.
    The constant factors can also work as a padding for the dimensionality to remain constant. '''

def _get_flat(x, keep_dim=False):
    if keep_dim:
        return x.reshape(torch.Size([1, x.size(0) * x.size(1)]) + x.size()[2:])
    return x.reshape(torch.Size([x.size(0) * x.size(1)]) + x.size()[2:])

# If we want it to be variational:
def _sample_latent_simple(mu, logvar, n_samples=1):
    std = torch.exp(0.5 * logvar)
    # std = torch.stack([std]*n_samples, dim=-1)
    eps = torch.randn_like(std)
    sample = mu + eps * std
    return sample

def _sample_latent_general(mu, var):
    std = var
    eps = torch.randn_like(std)
    return mu + var * eps

def logvar_to_matrix_var(logvar):
    var = torch.exp(logvar)
    var_mat = torch.diag_embed(var)
    return var_mat

class RecKoopmanModel(BaseModel):
    def __init__(self, in_channels, feat_dim, nf_particle, nf_effect, g_dim, u_dim,
                 n_objects, I_factor=10, n_blocks=1, psteps=1, n_timesteps=1, ngf=8, image_size=[64, 64]):
        super().__init__()
        out_channels = 1
        n_layers = int(np.log2(image_size[0])) - 1

        self.u_dim = u_dim

        # Temporal encoding buffers
        if n_timesteps > 1:
            t = torch.linspace(-1, 1, n_timesteps)
            # Add as constant, with extra dims for N and C
            self.register_buffer('t_grid', t)

        # Set state dim with config, depending on how many time-steps we want to take into account
        self.image_size = image_size
        self.n_timesteps = n_timesteps
        self.state_dim = feat_dim
        self.I_factor = I_factor
        self.psteps = psteps
        self.g_dim = g_dim
        self.n_objects = n_objects

        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

        self.linear_g = nn.Linear(g_dim, g_dim)
        self.linear_u = nn.Linear(g_dim, u_dim)
        self.initial_conditions = nn.Sequential(nn.Linear(feat_dim * n_timesteps * 2, feat_dim * n_timesteps),
                                                nn.ReLU(),
                                                nn.Linear(feat_dim * n_timesteps, g_dim * 2))

        self.image_encoder = ImageEncoder(in_channels, feat_dim * 2, n_objects, ngf, n_layers)  # feat_dim * 2 if sample here
        self.image_decoder = ImageDecoder(g_dim, out_channels, ngf, n_layers)
        self.koopman = KoopmanOperators(feat_dim * 2, nf_particle * 2, nf_effect * 2, g_dim, u_dim, n_timesteps, n_blocks)

        #Object attention:
        # if n_objects > 1:
        #     self.obj_attention = ObjectAttention(in_channels, feat_dim, n_objects)

    def _get_full_state(self, x, T):

        if self.n_timesteps < 2:
            return x, T
        new_T = T - self.n_timesteps + 1
        x = x.reshape(-1, T, *x.shape[1:])
        new_x = []
        for t in range(new_T):
            new_x.append(torch.cat([x[:, t + idx]
                         for idx in range(self.n_timesteps)], dim=-1))
        # torch.cat([ torch.zeros_like( , x[:,0,0:1]) + self.t_grid[idx]], dim=-1)
        new_x = torch.stack(new_x, dim=2)

        return new_x.reshape(-1, *new_x.shape[3:]), new_T

    def forward(self, input):
        bs, T, ch, h, w = input.shape

        f = self.image_encoder(input)
        # Concatenates the features from current time and previous to create full state
        f, T = self._get_full_state(f, T)
        # TODO: Might be bug. I'm reshaping with objects before T.
        f = f.view(torch.Size([bs * self.n_objects, T]) + f.size()[1:])

        f_mu, f_logvar, a_mu, a_logvar = [], [], [], []
        gs, us = [], []
        for t in range(T):
            if t==0:
                prev_sample = torch.zeros_like(f[:, 0, :self.g_dim])
                f_t = torch.cat([f[:, t], prev_sample], dim=-1)
                g = self.koopman.to_g(f_t, self.psteps)
                # g = self.initial_conditions(f[:, t])
            else:
                f_t = torch.cat([f[:, t], prev_sample], dim=-1)
                g = self.koopman.to_g(f_t, self.psteps)
                # TODO: provar recurrent, suposo
                # g, prev_hidden, u = self.koopman.to_g(f_t, prev_hidden) #,psteps=self.psteps
            g_mu, g_logvar = torch.chunk(g, 2, dim=-1)
            g = _sample_latent_simple(g_mu, g_logvar)

            # if t < T-1:
            #     u_mu, u_logvar = torch.chunk(u, 2, dim=-1)
            #     u = _sample_latent_simple(u_mu, u_logvar)
            # else:
            #     u_mu, u_logvar = torch.chunk(torch.zeros_like(u), 2, dim=-1)
            #     u = u_mu
            # g, u = self.linear_g(g), self.sigmoid(self.linear_u(u))
            g, u = self.linear_g(g), self.sigmoid(self.linear_u(g))

            prev_sample = g

            gs.append(g)
            us.append(u)

            f_mu.append(g_mu)
            f_logvar.append(g_logvar)
            # a_mu.append(u_mu)
            # a_logvar.append(u_logvar)

        g = torch.stack(gs, dim=1)
        f_mu = torch.stack(f_mu, dim=1)
        f_logvar = torch.stack(f_logvar, dim=1)

        u = torch.stack(us, dim=1)
        # u_zeros = torch.zeros_likes_like(u)
        # u = torch.where(u > 0.8, u, u_zeros)
        # a_mu = torch.stack(a_mu, dim=1)
        # a_logvar = torch.stack(a_logvar, dim=1)

        free_pred = 0
        if free_pred > 0:
            G_tilde = g[:, 1:-1-free_pred, None]  # new axis corresponding to N number of objects
            H_tilde = g[:, 2:-free_pred, None]
        else:
            G_tilde = g[:, 1:-1, None]  # new axis corresponding to N number of objects
            H_tilde = g[:, 2:, None]

        A, B, fit_err = self.koopman.system_identify(G=G_tilde, H=H_tilde, U=u[:, 1:-1], I_factor=self.I_factor)
        # TODO: clamp A before invert. Threshold u over 0.9

        # A, A_pinv, fit_err = self.koopman.fit_block_diagonal_A(G=G_tilde, H=H_tilde, I_factor=self.I_factor)
        # TODO: Try simulating backwards
        # B=None
        # Rollout. From observation in time 0, predict with Koopman operator
        # G_for_pred = self.koopman.simulate(T=T - 1, g=g[:, 0], u=u, A=A, B=B)
        # G_for_pred = self.koopman.simulate(T=T - 1, g=g[:, 0], u=None, A=A, B=None)

        '''Simple version of the reverse rollout'''
        # G_for_pred_rev = self.koopman.simulate(T=T - 1, g=g[:, 0], u=None, A=A_pinv, B=None)
        # G_for_pred = torch.flip(G_for_pred_rev, dims=[1])

        G_for_pred = torch.cat([g[:,0:1],self.koopman.simulate(T=T - 2, g=g[:, 1], u=u, A=A, B=B)], dim=1)


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

        returned_g = torch.cat([g, G_for_pred], dim=1)
        returned_mus = torch.cat([f_mu], dim=-1)
        returned_logvars = torch.cat([f_logvar], dim=-1)
        # returned_mus = torch.cat([f_mu, a_mu], dim=-1)
        # returned_logvars = torch.cat([f_logvar, a_logvar], dim=-1)

        o_touple = (out_rec, out_pred, returned_g.reshape(-1, returned_g.size(-1)),
                    returned_mus.reshape(-1, returned_mus.size(-1)),
                    returned_logvars.reshape(-1, returned_logvars.size(-1)))
                    # f_mu.reshape(-1, f_mu.size(-1)),
                    # f_logvar.reshape(-1, f_logvar.size(-1)))
        o = [item.reshape(torch.Size([bs * self.n_objects, -1]) + item.size()[1:]) for item in o_touple]
        # Option 1: one object mapped to 0
        # shape_o0 = o[0].shape
        # o[0] = o[0].reshape(bs, self.n_objects, *o[0].shape[1:])
        # o[0][:,0] = o[0][:,0]*0
        # o[0] = o[0].reshape(*shape_o0)

        # o[:2] = [torch.clamp(torch.sum(item.reshape(bs, self.n_objects, *item.shape[1:]), dim=1), min=0, max=1) for item in o[:2]]
        o[:2] = [torch.sum(item.reshape(bs, self.n_objects, *item.shape[1:]), dim=1) for item in o[:2]]
        o[3:5] = [item.reshape(bs, self.n_objects, *item.shape[1:]) for item in o[3:5]]

        o.append(A)
        o.append(B)
        o.append(u.reshape(bs * self.n_objects, -1, u.shape[-1]))
        # # Show images
        # plt.imshow(input[0, 0, :].reshape(16, 16).cpu().detach().numpy())
        # plt.savefig('test_attention.png')
        return o

class ConvAttentionKoopmanModel(BaseModel):
    def __init__(self, in_channels, feat_dim, nf_particle, nf_effect, g_dim,
                 n_objects, I_factor=10, psteps=1, n_timesteps=1, ngf=8, image_size=[64, 64]):
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

        # Temporal encoding buffers
        if n_timesteps > 1:
            t = torch.linspace(-1, 1, n_timesteps)
            # Add as constant, with extra dims for N and C
            self.register_buffer('t_grid', t)

        # Set state dim with config, depending on how many time-steps we want to take into account
        self.image_size = image_size
        self.n_timesteps = n_timesteps
        self.state_dim = feat_dim
        self.I_factor = I_factor
        self.psteps = psteps
        self.g_dim = g_dim
        self.n_objects = n_objects

        # self.att_image_encoder = AttImageEncoder(in_channels, feat_dim, ngf, n_layers)
        self.image_encoder = ImageEncoder(in_channels, feat_dim * 4, n_objects, ngf, n_layers)  # feat_dim * 2 if sample here
        self.image_decoder = ImageDecoder(g_dim, out_channels, ngf, n_layers)
        # self.image_decoder = SimpleSBImageDecoder(feat_dim, out_channels, ngf, n_layers, image_size)
        self.koopman = KoopmanOperators(self.state_dim * 4, nf_particle, nf_effect, g_dim, n_timesteps)

        #Object attention:
        if n_objects > 1:
            self.obj_attention = ObjectAttention(in_channels, feat_dim, n_objects)

    def _add_positional_encoding(self, x):

        x = torch.cat((self.x_grid_enc.expand(*x.shape[:2], -1, -1, -1),
                       self.y_grid_enc.expand(*x.shape[:2], -1, -1, -1),
                       x), dim=-3)

        return x

    def _get_full_state(self, x, T):

        if self.n_timesteps < 2:
            return x, T
        new_T = T - self.n_timesteps + 1
        x = x.reshape(-1, T, *x.shape[1:])
        new_x = []
        for t in range(new_T):
            new_x.append(torch.cat([x[:, t + idx]
                         for idx in range(self.n_timesteps)], dim=-1))
        # torch.cat([ torch.zeros_like( , x[:,0,0:1]) + self.t_grid[idx]], dim=-1)
        new_x = torch.stack(new_x, dim=1)
        return new_x.reshape(-1, *new_x.shape[2:]), new_T

    def forward(self, input):
        bs, T, ch, h, w = input.shape

        #Object attention
        if self.n_objects > 1:
            o_a = self.obj_attention(input.permute(0,2,1,3,4)).reshape(bs, self.n_objects, T, 1, h, w)
            input = input[:,None]*o_a

        # input = self._add_positional_encoding(input)
        f = self.image_encoder(input.reshape(-1, ch, h, w))


        # # Concatenates the features from current time and previous to create full state
        f, T = self._get_full_state(f, T)
        # f = f.view(torch.Size([bs, T]) + f.size()[1:])
        # g, G_for_pred = f, f

        g = self.koopman.to_g(f, self.psteps)
        g = g.view(torch.Size([bs * self.n_objects, T]) + g.size()[1:])

        free_pred = 0 # TODO: Set it to >1 after training for several epochs.
        if free_pred > 0:
            G_tilde = g[:, :-1-free_pred, None]  # new axis corresponding to N number of objects
            H_tilde = g[:, 1:-free_pred, None]
        else:
            G_tilde = g[:, :-1, None]  # new axis corresponding to N number of objects
            H_tilde = g[:, 1:, None]

        # Find Koopman operator from current data (matrix a)
        # TODO: Predict longer from an A obtained for a shorter length.
        A, fit_err = self.koopman.system_identify(G=G_tilde, H=H_tilde, I_factor=self.I_factor)

        # Rollout. From observation in time 0, predict with Koopman operator
        G_for_pred = self.koopman.simulate(T=T - 1, g=g[:, 0], A=A)  # Rollout from initial observation to the last

        # # Note: Ignore - If we have split the representation: Merge constant and dynamic components
        # if g.size(-1) < self.g_dim:
        #     g = torch.cat([g, g_cte[:, 0:1].repeat(1, T, 1)], dim=-1)
        #     G_for_pred = torch.cat([G_for_pred, g_cte[:, 0:1].repeat(1, T - 1, 1)], dim=-1)


        g_mu, g_logvar = torch.chunk(g, 2, dim=-1)
        g_mu_pred, g_logvar_pred = torch.chunk(G_for_pred, 2, dim=-1)
        g = _sample_latent_simple(g_mu, g_logvar)
        G_for_pred = _sample_latent_simple(g_mu_pred, g_logvar_pred)
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

        returned_mus = torch.cat([f_mu, g_mu_pred], dim=1)
        returned_logvars = torch.cat([f_logvar, g_logvar_pred], dim=1)
        o_touple = (out_rec, out_pred, returned_mus.reshape(-1, returned_mus.size(-1)),
                    returned_mus.reshape(-1, returned_mus.size(-1)),
                    returned_logvars.reshape(-1, returned_logvars.size(-1)))
                    # f_mu.reshape(-1, f_mu.size(-1)),
                    # f_logvar.reshape(-1, f_logvar.size(-1)))
        o = [item.reshape(torch.Size([bs * self.n_objects, -1]) + item.size()[1:]) for item in o_touple]
        # o[:2] = [torch.clamp(torch.sum(item.reshape(bs, self.n_objects, *item.shape[1:]), dim=1), min=0, max=1.5) for item in o[:2]]
        o[:2] = [torch.sum(item.reshape(bs, self.n_objects, *item.shape[1:]), dim=1) for item in o[:2]]

        # A_to_show = self.block_diagonal(A)
        o.append(A[..., :self.g_dim, :self.g_dim])
        if self.n_objects > 1:
            o.append(o_a)
        # Returns reconstruction, prediction, g_representation, mu and sigma from which we sample to compute KL divergence

        # # Show images
        # plt.imshow(input[0, 0, :].reshape(16, 16).cpu().detach().numpy())
        # plt.savefig('test_attention.png')
        return o

class SingleObjKoopmanModel(BaseModel):
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

        # Temporal encoding buffers
        if n_timesteps > 1:
            t = torch.linspace(-1, 1, n_timesteps)
            # Add as constant, with extra dims for N and C
            self.register_buffer('t_grid', t)

        # Set state dim with config, depending on how many time-steps we want to take into account
        self.image_size = image_size
        self.n_timesteps = n_timesteps
        self.state_dim = feat_dim
        self.I_factor = I_factor
        self.psteps = psteps
        self.g_dim = g_dim

        # self.att_image_encoder = AttImageEncoder(in_channels, feat_dim, ngf, n_layers)
        self.image_encoder = ImageEncoder(in_channels, feat_dim * 4, ngf, n_layers)  # feat_dim * 2 if sample here
        self.image_decoder = ImageDecoder(g_dim, out_channels, ngf, n_layers)
        # self.image_decoder = SimpleSBImageDecoder(feat_dim, out_channels, ngf, n_layers, image_size)
        self.koopman = KoopmanOperators(self.state_dim * 4, nf_particle, nf_effect, g_dim, n_timesteps)

    def _add_positional_encoding(self, x):

        x = torch.cat((self.x_grid_enc.expand(*x.shape[:2], -1, -1, -1),
                       self.y_grid_enc.expand(*x.shape[:2], -1, -1, -1),
                       x), dim=-3)

        return x

    def _get_full_state(self, x, T):

        if self.n_timesteps < 2:
            return x, T
        new_T = T - self.n_timesteps + 1
        x = x.reshape(-1, T, *x.shape[1:])
        new_x = []
        for t in range(new_T):
            new_x.append(torch.cat([x[:, t + idx]
                         for idx in range(self.n_timesteps)], dim=-1))
        # torch.cat([ torch.zeros_like( , x[:,0,0:1]) + self.t_grid[idx]], dim=-1)
        new_x = torch.stack(new_x, dim=1)
        return new_x.reshape(-1, *new_x.shape[2:]), new_T

    def forward(self, input):
        bs, T, ch, h, w = input.shape

        # Note: test image encoder with attention
        # f_mu, f_logvar = torch.chunk(self.att_image_encoder(input.reshape(-1, ch, h, w)), 2, dim=-1)
        # f = _sample(f_mu, f_logvar)

        # Note: Koopman applied to mu and logvar
        # f = self.att_image_encoder(input.reshape(-1, ch, h, w))

        # input = self._add_positional_encoding(input)
        f = self.image_encoder(input.reshape(-1, ch, h, w)) #Note: deterministic AE
        # f_mu, f_logvar = f, f
        # f_mu, f_logvar = torch.chunk(self.image_encoder(input.reshape(-1, ch+2, h, w)), 2, dim=-1)
        # f = _sample(f_mu, f_logvar)
        # f = f_mu

        # # Concatenates the features from current time and previous to create full state
        f, T = self._get_full_state(f, T)
        # f = f.view(torch.Size([bs, T]) + f.size()[1:])
        # g, G_for_pred = f, f

        g = self.koopman.to_g(f, self.psteps)
        g = g.view(torch.Size([bs, T]) + g.size()[1:])
        # g = f.view(torch.Size([bs, T]) + f.size()[1:])

        # # Note: Ignore - Split representation into constant and dynamic by hardcoding
        # g, g_cte = torch.chunk(g, 2, dim=-1) # Option 1
        # g, g_cte = g[..., :2], g[..., 2:] # Option 2

        free_pred = 3
        G_tilde = g[:, :-1-free_pred, None]  # new axis corresponding to N number of objects
        H_tilde = g[:, 1:-free_pred, None]

        # Find Koopman operator from current data (matrix a)
        # TODO: Predict longer from an A obtained for a shorter length.
        A, fit_err = self.koopman.system_identify(G=G_tilde, H=H_tilde, I_factor=self.I_factor)

        # Rollout. From observation in time 0, predict with Koopman operator
        G_for_pred = self.koopman.simulate(T=T - 1, g=g[:, 0], A=A)  # Rollout from initial observation to the last

        # # Note: Ignore - If we have split the representation: Merge constant and dynamic components
        # if g.size(-1) < self.g_dim:
        #     g = torch.cat([g, g_cte[:, 0:1].repeat(1, T, 1)], dim=-1)
        #     G_for_pred = torch.cat([G_for_pred, g_cte[:, 0:1].repeat(1, T - 1, 1)], dim=-1)


        g_mu, g_logvar = torch.chunk(g, 2, dim=-1)
        g_mu_pred, g_logvar_pred = torch.chunk(G_for_pred, 2, dim=-1)
        g = _sample_latent_simple(g_mu, g_logvar)
        G_for_pred = _sample_latent_simple(g_mu_pred, g_logvar_pred)
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

        # # Show images
        # plt.imshow(input[0, 0, :].reshape(16, 16).cpu().detach().numpy())
        # plt.savefig('test_attention.png')
        return o

class svae_KoopmanModel(BaseModel):
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

        ## x coordinate array
        xgrid = np.linspace(-1, 1, image_size[0])
        ygrid = np.linspace(1, -1, image_size[1])
        x0, x1 = np.meshgrid(xgrid, ygrid)
        x_coord = np.stack([x0.ravel(), x1.ravel()], 1)
        self.register_buffer('x_coord', torch.from_numpy(x_coord).float())

        # Set state dim with config, depending on how many time-steps we want to take into account
        self.image_size = image_size
        self.n_timesteps = n_timesteps
        self.state_dim = feat_dim
        self.I_factor = I_factor
        self.psteps = psteps
        self.g_dim = g_dim

        # Priors
        self.theta_prior = np.pi
        self.dx_scale = 0.1

        # self.image_encoder = ImageEncoder(in_channels + 2, feat_dim * 2, ngf, n_layers)  # feat_dim * 2 if sample here
        self.image_encoder = LinearEncoder(self.image_size[0]*self.image_size[1], feat_dim, feat_dim * 4, num_layers=1) #n, latent_dim, hidden_dim, num_layers=1, activation=nn.Tanh, resid=False):
        self.koopman = KoopmanOperators(self.state_dim, nf_particle, nf_effect, g_dim, n_timesteps)
        # self.image_decoder = ImageDecoder(feat_dim, out_channels, ngf, n_layers)
        self.image_decoder = linearSpatialDecoder(feat_dim - 2, out_channels, feat_dim, ngf, num_layers=1) # -3 coord part

    def _add_positional_encoding(self, x):

        x = torch.cat((self.x_grid_enc.expand(*x.shape[:2], -1, -1, -1),
                       self.y_grid_enc.expand(*x.shape[:2], -1, -1, -1),
                       x), dim=-3)

        return x

    def _get_full_state(self, x, T):

        if self.n_timesteps < 2:
            return x, T
        new_T = T - self.n_timesteps + 1
        x = x.reshape(-1, T, *x.shape[1:])
        new_x = []
        for t in range(new_T):
            new_x.append(torch.cat([x[:, t + idx]
                         for idx in range(self.n_timesteps)], dim=-1))
        # torch.cat([ torch.zeros_like( , x[:,0,0:1]) + self.t_grid[idx]], dim=-1)
        new_x = torch.stack(new_x, dim=1)
        return new_x.reshape(-1, *new_x.shape[2:]), new_T

    def forward(self, input):
        bs, T, ch, h, w = input.shape

        # input = self._add_positional_encoding(input)
        # f = self.image_encoder(input.reshape(-1, ch + 2, h, w))
        f = self.image_encoder(input.reshape(-1, ch, h, w))

        # Concatenates the features from current time and previous to create full state
        # f, T = self._get_full_state(f, T) # TODO: add positional encoding [0, 1]

        # f = f[..., None, None]
        # g = self.koopman.to_g(f, self.psteps)
        # g = g.view(torch.Size([bs, T]) + g.size()[1:])
        g = f.view(torch.Size([bs, T]) + f.size()[1:])

        G_tilde = g[:, :-1, None]  # new axis corresponding to N number of objects
        H_tilde = g[:, 1:, None]

        # Find Koopman operator from current data (matrix a)
        A, fit_err = self.koopman.system_identify(G=G_tilde, H=H_tilde, I_factor=self.I_factor)

        # Rollout. From observation in time 0, predict with Koopman operator
        G_for_pred = self.koopman.simulate(T=T - 1, g=g[:, 0], A=A)  # Rollout from initial observation to the last

        g_mu, g_logvar = torch.chunk(g, 2, dim=-1)
        g_mu_pred, g_logvar_pred = torch.chunk(G_for_pred, 2, dim=-1)
        g, x_coord, kl_div = self._sample_latent_spatial(g_mu, g_logvar)
        G_for_pred, x_coord_for_pred, _ = self._sample_latent_spatial(g_mu_pred, g_logvar_pred)
        f_mu, f_logvar = g_mu, g_logvar

        # Option 1: use the koopman object decoder
        # s_for_rec = self.koopman.to_s(gcodes=_get_flat(g),
        #                               pstep=self.psteps)
        # s_for_pred = self.koopman.to_s(gcodes=_get_flat(torch.cat([g[:, :1],G_for_pred], dim=1)),
        #                                pstep=self.psteps)
        # Option 2: we don't use the koopman object decoder
        # s_for_rec = _get_flat(g)
        # s_for_pred = _get_flat(G_for_pred)
        # Option 3: no koopman decoder, no flattening
        s_for_rec = g
        s_for_pred = G_for_pred

        # Convolutional decoder. Normally Spatial Broadcasting decoder
        out_rec = self.image_decoder(x_coord, s_for_rec)
        out_pred = self.image_decoder(x_coord_for_pred, s_for_pred)

        o_touple = (out_rec, out_pred, f_mu.reshape(-1, f_mu.size(-1)))
        o = [item.reshape(torch.Size([bs, -1]) + item.size()[1:]) for item in o_touple]
        o.append(kl_div)
        # Returns reconstruction, prediction, g_representation, mu and sigma from which we sample to compute KL divergence

        return o

    def _sample_latent_spatial(self, z_mu, z_logstd, rotate=False, translate=True):
        z_mu, z_logstd = z_mu.reshape(-1, z_mu.shape[-1]), \
                         z_logstd.reshape(-1, z_mu.shape[-1])
        b = z_mu.size(0)

        x_coord = self.x_coord.expand(b, self.x_coord.size(0), self.x_coord.size(1))

        z_std = torch.exp(z_logstd)
        z_dim = z_mu.size(1)

        # draw samples from variational posterior to calculate
        # E[p(x|z)]
        r = Variable(x_coord.data.new(b, z_dim).normal_())
        z = z_std * r + z_mu

        kl_div = 0
        if rotate:
            # z[0] is the rotation
            theta_mu = z_mu[:, 0]
            theta_std = z_std[:, 0]
            theta_logstd = z_logstd[:, 0]
            theta = z[:, 0]
            z = z[:, 1:]
            z_mu = z_mu[:, 1:]
            z_std = z_std[:, 1:]
            z_logstd = z_logstd[:, 1:]

            # calculate rotation matrix
            rot = Variable(theta.data.new(b, 2, 2).zero_())
            rot[:, 0, 0] = torch.cos(theta)
            rot[:, 0, 1] = torch.sin(theta)
            rot[:, 1, 0] = -torch.sin(theta)
            rot[:, 1, 1] = torch.cos(theta)
            x_coord = torch.bmm(x_coord, rot)  # rotate coordinates by theta

            # calculate the KL divergence term
            sigma = self.theta_prior
            kl_div = -theta_logstd + np.log(sigma) + (theta_std ** 2 + theta_mu ** 2) / 2 / sigma ** 2 - 0.5

        if translate:
            # z[0,1] are the translations
            # dx_mu = z_mu[:, :2]
            # dx_std = z_std[:, :2]
            # dx_logstd = z_logstd[:, :2]
            dx = z[:, :2] * self.dx_scale  # scale dx by standard deviation
            dx = dx.unsqueeze(1)
            z = z[:, 2:]

            x_coord = x_coord + dx  # translate coordinates

        # unit normal prior over z and translation
        z_kl = -z_logstd + 0.5 * z_std ** 2 + 0.5 * z_mu ** 2 - 0.5
        kl_div = kl_div + torch.sum(z_kl, 1)
        kl_div = kl_div.mean()

        return z, x_coord, kl_div

class cswm_KoopmanModel(BaseModel):
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
            input_dim=in_channels,
            hidden_dim=feat_dim // 8,
            output_dim=feat_dim,
            num_objects=num_objects)

        self.obj_encoder = EncoderMLP(
            input_dim=np.prod(image_size) // (8 ** 2) * feat_dim,
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
            new_x.append(torch.cat([x[:, t + idx] for idx in range(self.n_timesteps)], dim=-1))

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

        G_tilde = g[:, :-1, None]  # new axis corresponding to N number of objects
        H_tilde = g[:, 1:, None]

        A, fit_err = self.koopman.system_identify(G=G_tilde, H=H_tilde, I_factor=self.I_factor)  # TODO: maybe

        ''' rollout: BT x D '''
        G_for_pred = self.koopman.simulate(T=T - 1, g=g[:, 0], A=A)  # Rollout from initial observation to the last

        s_for_rec = self.koopman.to_s(gcodes=_get_flat(g), pstep=self.psteps)
        s_for_pred = self.koopman.to_s(gcodes=_get_flat(G_for_pred), pstep=self.psteps)

        out_rec = self.image_decoder(s_for_rec)
        out_pred = self.image_decoder(s_for_pred)

        o_touple = (out_rec, out_pred, g.reshape(-1, g.size(-1)), f_mu.reshape(-1, f_mu.size(-1)),
                    f_logvar.reshape(-1, f_logvar.size(-1)))
        o = [item.reshape(torch.Size([bs, -1]) + item.size()[1:]) for item in o_touple]

        return o