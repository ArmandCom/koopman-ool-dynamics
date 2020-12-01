import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from base import BaseModel
# from model.networks import ImageEncoder, SBImageDecoder, SimpleSBImageDecoder, \
#     KoopmanOperators, AttImageEncoder, ImageDecoder, \
#     deconvSpatialDecoder, linearSpatialDecoder, LinearEncoder, ObjectAttention
from model.networks import KoopmanOperators
from model.networks_slot_attention import ImageEncoder, ImageDecoder
# from model.networks_cswm.modules import TransitionGNN, EncoderCNNLarge, EncoderCNNMedium, EncoderCNNSmall, EncoderMLP, DecoderCNNLarge, DecoderCNNMedium, DecoderCNNSmall, DecoderMLP
from torch.autograd import Variable
from torch.distributions.categorical import Categorical
import random
import matplotlib.pyplot as plt

''' A main objective of the model is to separate the dimensions to be modeled/propagated by koopman operator
    from the constant dimensions - elements that remain constant - which should be the majority of info. 
    This way we only need to control a reduced amount of dimensions - and we can make them variable.
    The constant factors can also work as a padding for the dimensionality to remain constant. '''

#Note: preguntes Octavia:
# Can we minimize hankel rank including input?
# I can't manage to correctly predict u. How should it look like?
# Lack of intuition with koopman. Should the same matrix apply to all?
#   Different directions use the same matrix or the inverse.
#   If it's the same, it means the matrix is orthogonal. Is it also symmetric?
# We have successfully separated the constant from the varying content and objects from each other.
# I have the poster for DIVE
# Idea

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

def _clamp_diagonal(A, min, max):
    # Note: this doesn't help. We need spectral normalization.
    eye = torch.zeros_like(A)
    ids = torch.arange(0, A.shape[-1])
    eye[..., ids, ids] = 1
    return A*(1-eye) + torch.clamp(A * eye, min, max)

'''Gumble sampling'''
# def _sample_gumbel(logits, eps=1e-20):
#     # U = torch.rand(shape).cuda()
#     U = torch.rand_like(logits)
#     return -Variable(torch.log(-torch.log(U + eps) + eps))
#
# def _gumbel_softmax_sample(logits, temperature):
#     # y = logits + _sample_gumbel(logits.size())
#     y = logits + _sample_gumbel(logits)
#     return F.softmax(y / temperature, dim=-1)
#
# def _gumbel_softmax(logits, temperature=1):
#     """
#     ST-gumple-softmax
#     input: [*, n_class]
#     return: flatten --> [*, n_class] an one-hot vector
#     """
#     latent_dim = logits.shape[-1]//2
#     y = _gumbel_softmax_sample(logits, temperature)
#     shape = y.size()
#     _, ind = y.max(dim=-1)
#     y_hard = torch.zeros_like(y).view(-1, shape[-1])
#     y_hard.scatter_(1, ind.view(-1, 1), 1)
#     y_hard = y_hard.view(*shape)
#     y_hard = (y_hard - y).detach() + y
#     return y_hard.view(-1,latent_dim*2)


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

        # self.softmax = nn.Softmax(dim=-1)
        # self.sigmoid = nn.Sigmoid()

        # self.linear_g = nn.Linear(g_dim, g_dim)

        # self.initial_conditions = nn.Sequential(nn.Linear(feat_dim * n_timesteps * 2, feat_dim * n_timesteps),
        #                                         nn.ReLU(),
        #                                         nn.Linear(feat_dim * n_timesteps, g_dim * 2))

        feat_dyn_dim = feat_dim // 6
        self.feat_dyn_dim = feat_dyn_dim
        self.content = None
        # self.reverse = True
        self.reverse = False
        self.ini_alpha = 1
        self.incr_alpha = 0.25

        self.linear_f_mu = nn.Linear(feat_dim * 2, feat_dim)
        self.linear_f_logvar = nn.Linear(feat_dim * 2, feat_dim)

        self.image_encoder = ImageEncoder(in_channels, feat_dim * 2, n_objects, ngf, n_layers)  # feat_dim * 2 if sample here
        self.image_decoder = ImageDecoder(feat_dim, out_channels, ngf, n_layers)
        self.koopman = KoopmanOperators(feat_dyn_dim, nf_particle, nf_effect, g_dim, u_dim, n_timesteps, n_blocks)

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
        return new_x.reshape(-1, new_x.shape[-1]), new_T

    def _get_full_state_hankel(self, x, T):
        '''
        :param x: features or observations
        :param T: number of time-steps before concatenation
        :return: Columns of a hankel matrix with self.n_timesteps rows.
        '''
        if self.n_timesteps < 2:
            return x, T
        new_T = T - self.n_timesteps + 1

        x = x.reshape(-1, T, *x.shape[2:])
        new_x = []
        for t in range(new_T):
            new_x.append(torch.stack([x[:, t + idx]
                                    for idx in range(self.n_timesteps)], dim=-1))
        # torch.cat([ torch.zeros_like( , x[:,0,0:1]) + self.t_grid[idx]], dim=-1)
        new_x = torch.stack(new_x, dim=1)

        return new_x.reshape(-1, new_T, new_x.shape[-2] * new_x.shape[-1]), new_T


    def forward(self, input, epoch = 1):
        bs, T, ch, h, w = input.shape
        n_dirs = 1
        f = self.image_encoder(input) # (bs * n_objects * T, feat_dim)
        f_mu = self.linear_f_mu(f)
        f_logvar = self.linear_f_logvar(f)
        # Concatenates the features from current time and previous to create full state
        # f_mu, f_logvar = torch.chunk(f, 2, dim=-1)
        f = _sample_latent_simple(f_mu, f_logvar)

        if self.reverse:
            f = f.reshape(bs, T, *f.shape[1:])
            f_rev = torch.flip(f, dims=[1])
            f = torch.stack([f, f_rev], dim=1)
            bs, n_dirs = 2*bs, 2
            f = f.reshape(bs * T, f.shape[-1]) # (bs * n_objects * n_dir * T, feat_dim) And from this point on bs doubles.

        # Note: Split features (cte, dyn)
        f_dyn, f_cte = f[..., :self.feat_dyn_dim], f[..., self.feat_dyn_dim:]
        f_cte = f_cte.reshape(bs * self.n_objects, T, *f_cte.shape[1:])

        # Test disentanglement
        # if random.random() < 0.1 or self.content is None:
        #     self.content = f_cte
        # f_cte = self.content

        f = f_dyn.reshape(bs * self.n_objects, T, *f_dyn.shape[1:])
        f, T = self._get_full_state_hankel(f, T)

        u, u_dist = self.koopman.to_u(f, temp=self.ini_alpha + epoch * self.incr_alpha)
        g = self.koopman.to_g(f.reshape(bs * self.n_objects * T, -1), self.psteps)
        g = g.reshape(bs * self.n_objects, T, *g.shape[1:])

        if self.reverse:
            bs = bs//n_dirs
            g = g.reshape(bs, self.n_objects, n_dirs, T, *g.shape[2:])
            g_fw = g[:, :, 0].reshape(bs * self.n_objects, T, *g.shape[4:])
            g_bw = g[:, :, 1].reshape(bs * self.n_objects, T, *g.shape[4:])
            g = g_bw

        # TODO:
        #  0: state as velocity acceleration, pose
        #  0: g fitting error.
        #  0: Hankel view. Fuck koopman for now, it has too many degrees of freedom. Restar input para minimizar rango de H.
        #  1: Invert or Sample randomly u and maximize the error in reconstruction.
        #  2: Treat n_timesteps with conv_nn? Or does it make sense to mix f1(t) and f2(t-1)?
        #  3: We don't impose A symmetric, but we impose that g in both directions use the same A and B.
        #       Explode symmetry. If there's input in the a sequence, there will be the same input in the reverse sequence.
        #       B is valid for both. Then we cant cross, unless we recalculate B. It might not be necessary because we use
        #       the same A for both dirs. INVERT U's temporally --> that's a great idea. Dismiss (n_ini_timesteps - 1 samples) in each extreme Is this correct?
        #       Is it the same linear input intervention if I do it forward and backward?
        #  5: Eigenvalues and Eigenvector. Fix one get the other.
        #  6: Observation selector using Gumbel (should apply mask emissor and receptor
        #       and it should be the same across time)
        #  7: Attention mechanism to select - koopman (sub-koopman) / Identity. It will be useful for the future.
        #  8: When works, formalize code. Hyperparameters through config.yaml
        #  9: The nonlinear projection should also separate the objects. We can separate by the svd? Is it possible?
        # 1: Prev_sample not necessary? Start from g(0)
        # 2: Sampling from features
        # 3: Bottleneck in f. Smaller than G. Constant part is big and routed directly.
        #       Variant part is very small and expanded to G.
        # 4: In evaluation, sample features several times but only keep one of the cte. Check if it captures the appearance
        # 5: Fix B with A learned. If the first doesn't require input, B will be mapped to 0.

        randperm = torch.arange(g.shape[0])  if self.reverse or True\
            else torch.randperm(g.shape[0])

        #Note: Inverting u(t) in the time axis
        # TODO: still here but im tired.
        if self.reverse:
            u_fw = u
            zeros = torch.zeros_like(u[:, :self.n_timesteps])
            u_bw = torch.flip(u, dims=[1])
            u_bw = torch.cat([u_bw[:, self.n_timesteps:], zeros], dim=1)
            u = u_bw
            free_pred = self.n_timesteps - 1 + 4
        else:
            free_pred = T//4

        # if free_pred > 0:
        #     G_tilde = g[randperm, self.n_timesteps -1:-1-free_pred, None]
        #     H_tilde = g[randperm, self.n_timesteps:-free_pred, None]
        # else:
        #     G_tilde = g[randperm, self.n_timesteps -1:-1, None]
        #     H_tilde = g[randperm, self.n_timesteps:, None]

        if free_pred > 0:
            G_tilde = g[randperm, :-1-free_pred, None]
            H_tilde = g[randperm, 1:-free_pred, None]
        else:
            G_tilde = g[randperm, :-1, None]
            H_tilde = g[randperm, 1:, None]

        # TODO: If we identify with half of the timesteps, but use all inputs for rollout,
        #  we might get something. We can also predict only the future.
        A, B, A_inv, fit_err = self.koopman.system_identify(G=G_tilde, H=H_tilde, U=u[randperm, :-1-free_pred], I_factor=self.I_factor)
        # A, B, A_inv, fit_err = self.koopman.fit_with_A(G=G_tilde, H=H_tilde, U=u[randperm, :-1], I_factor=self.I_factor)
        # A, B, A_inv, fit_err = self.koopman.fit_with_B(G=G_tilde, H=H_tilde, U=u[randperm, :-1-free_pred], I_factor=self.I_factor)
        # A, B = self.koopman.fit_with_AB(G_tilde.shape[0])
        # A, A_pinv, fit_err = self.koopman.fit_block_diagonal_A(G=G_tilde, H=H_tilde, I_factor=self.I_factor)
        # B = None

        if self.reverse:
            g = g_fw
            u = u_fw
            f_cte_shape = f_cte.shape
            f_cte = f_cte.reshape(bs, n_dirs * f_cte_shape[1], *f_cte_shape[2:])

        # G_for_pred = self.koopman.simulate(T=T - 1, g=g[:, 0], u=u, A=A, B=B)
        # g_for_koop = self.koopman.simulate(T=T - 1, g=g[:, 0], u=None, A=A, B=None)
        # G_for_pred = torch.cat([g.reshape(*g.shape[:2], -1, self.n_timesteps)[:,1:self.n_timesteps, :, -1],
        #                         self.koopman.simulate(T=T - self.n_timesteps, g=g[:, self.n_timesteps -1], u=u[:, self.n_timesteps -1:], A=A, B=B)], dim=1)
        # g_for_koop = torch.cat([g.reshape(*g.shape[:2], -1, self.n_timesteps)[:,1:self.n_timesteps, :, -1],
        #                         self.koopman.simulate(T=T - self.n_timesteps, g=g[:, self.n_timesteps -1], u=None, A=A, B=None)], dim=1)
        # G_for_pred = self.koopman.simulate(T=T - self.n_timesteps, g=g[:, self.n_timesteps -1], u=u[:, self.n_timesteps -1:], A=A, B=B)
        # g_for_koop = self.koopman.simulate(T=T - self.n_timesteps, g=g[:, self.n_timesteps -1], u=None, A=A, B=None)
        G_for_pred = self.koopman.simulate(T=T - 4, g=g[:, 3], u=u, A=A, B=B)
        g_for_koop = self.koopman.simulate(T=T - 4, g=g[:, 3], u=None, A=A, B=None)


        # g_for_koop = G_for_pred

        # TODO: create recursive rollout. We obtain the input at each step from g
        # G_for_pred = self.koopman.simulate(T=T - 1, g=g[:, 0], u=None, A=A, B=None)
        '''Simple version of the reverse rollout'''
        # G_for_pred_rev = self.koopman.simulate(T=T - 1, g=g[:, 0], u=None, A=A_pinv, B=None)
        # G_for_pred = torch.flip(G_for_pred_rev, dims=[1])
        # G_for_pred = torch.cat([g[:,0:1],self.koopman.simulate(T=T - 2, g=g[:, 1], u=u, A=A, B=B)], dim=1)

        # if self.hankel:
        #     # G_for_pred = G_for_pred.reshape(*G_for_pred.shape[:-1],
        #     #                                 G_for_pred.shape[-1]// self.n_timesteps,
        #     #                                 self.n_timesteps)[..., self.n_timesteps - 1]
        #     # g = g.reshape(*g.shape[:-1],
        #     #               g.shape[-1]// self.n_timesteps,
        #     #               self.n_timesteps)[..., self.n_timesteps - 1]
        #     g_for_koop, T_prime = self._get_full_state_hankel(g_for_koop, g_for_koop.shape[1])
        #     g_for_koop = g_for_koop.reshape(*g_for_koop.shape[:-1],
        #                   g_for_koop.shape[-1]// self.n_timesteps,
        #                   self.n_timesteps)
        #     # #Option 2: with hankel structure.
        #     # G_for_pred = G_for_pred.reshape(*G_for_pred.shape[:-1],
        #     #                                 G_for_pred.shape[-1])
        #     # g = g.reshape(*g.shape[:-1],
        #     #               g.shape[-1]//self.n_timesteps,
        #     #               self.n_timesteps)[..., self.n_timesteps - 1]


        # Option 1: use the koopman object decoder
        s_for_rec = self.koopman.to_s(gcodes=_get_flat(g),
                                      psteps=self.psteps)
        s_for_pred = self.koopman.to_s(gcodes=_get_flat(G_for_pred),
                                       psteps=self.psteps)

        # Note: Split features (cte, dyn). In case of reverse, f_cte averages for both directions.
        # Note 2: TODO: Reconstruction could be in reversed g's or both!
        s_for_rec = torch.cat([s_for_rec.reshape(bs * self.n_objects, T, -1),
                              f_cte[:, None].mean(2).repeat(1, T, 1)], dim=-1)
        s_for_pred = torch.cat([s_for_pred.reshape(bs * self.n_objects, G_for_pred.shape[1], -1),
                              f_cte[:, None].mean(2).repeat(1, G_for_pred.shape[1], 1)], dim=-1)
        s_for_rec = _get_flat(s_for_rec)
        s_for_pred = _get_flat(s_for_pred)

        # Option 2: we don't use the koopman object decoder
        # s_for_rec = _get_flat(g)
        # s_for_pred = _get_flat(G_for_pred)

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

        # u = u.reshape(*u.shape[:2], self.u_dim, self.n_timesteps)[..., -1] #TODO:HANKEL This is for hankel view
        o.append(u.reshape(bs * self.n_objects, -1, u.shape[-1]))
        o.append(u_dist.reshape(bs * self.n_objects, -1, *u_dist.shape[-1:])) #Append udist for categorical
        # o.append(u_dist.reshape(bs * self.n_objects, -1, *u_dist.shape[-2:]))
        o.append(g_for_koop.reshape(bs * self.n_objects, -1, g_for_koop.shape[-1]// self.n_timesteps, self.n_timesteps))
        o.append(fit_err)

        return o