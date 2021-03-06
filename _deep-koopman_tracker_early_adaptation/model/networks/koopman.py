from abc import ABC

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from model.networks_space.spectral_norm import SpectralNorm

class RelationEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RelationEncoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        )

    def forward(self, x):
        """
        Args:
            x: [n_relations, input_size]
        Returns:
            [n_relations, output_size]
        """
        return self.model(x)


class ParticleEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ParticleEncoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        )

    def forward(self, x):
        """
        Args:
            x: [n_particles, input_size]
        Returns:
            [n_particles, output_size]
        """
        return self.model(x)


class Propagator(nn.Module):
    def __init__(self, input_size, output_size, residual=False):
        super(Propagator, self).__init__()

        self.residual = residual

        self.linear = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x, res=None):
        """
        Args:
            x: [n_relations/n_particles, input_size]
        Returns:
            [n_relations/n_particles, output_size]
        """
        if self.residual:
            x = self.relu(self.linear(x) + res)
        else:
            x = self.relu(self.linear(x))

        return x


class ParticlePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ParticlePredictor, self).__init__()

        self.linear_0 = nn.Linear(input_size, hidden_size)
        self.linear_1 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: [n_particles, input_size]
        Returns:
            [n_particles, output_size]
        """
        x = self.relu(self.linear_0(x))

        return self.linear_1(x)

class SNParticlePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SNParticlePredictor, self).__init__()

        self.linear_0 = SpectralNorm(nn.Linear(input_size, hidden_size))
        self.linear_1 = SpectralNorm(nn.Linear(hidden_size, output_size))
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: [n_particles, input_size]
        Returns:
            [n_particles, output_size]
        """
        x = self.relu(self.linear_0(x))

        return self.linear_1(x)

class ActionPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ActionPredictor, self).__init__()

        self.linear_0 = nn.Linear(input_size, hidden_size)
        self.linear_1 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: [n_particles, input_size]
        Returns:
            [n_particles, output_size]
        """
        x = self.relu(self.linear_0(x))

        return self.linear_1(x)


class l2_norm_layer(nn.Module):
    def __init__(self):
        super(l2_norm_layer, self).__init__()

    def forward(self, x):
        """
        :param x:  B x D
        :return:
        """
        norm_x = torch.sqrt((x ** 2).sum(1) + 1e-10)
        return x / norm_x[:, None]

# class RecurrentPropagationNetwork(nn.Module):
#
#     def __init__(self, input_particle_dim, nf_particle, nf_effect, output_dim, output_action_dim = None,
#                  tanh=False, residual=False, use_gpu=False):
#
#         super(RecurrentPropagationNetwork, self).__init__()
#
#         self.use_gpu = use_gpu
#         self.residual = residual
#         self.output_action_dim = output_action_dim
#         self.output_dim = output_dim
#
#         self.particle_predictor = ParticlePredictor(input_particle_dim, nf_particle, input_particle_dim)
#         self.action_predictor = ParticlePredictor(input_particle_dim, nf_effect, output_action_dim)
#
#         # self.t_enc = nn.GRU(input_particle_dim, input_particle_dim, 2, batch_first=True)
#         # self.t_trans = nn.GRU(input_particle_dim, output_dim, 2, batch_first=True)
#         self.t_trans = nn.GRUCell(input_particle_dim, output_dim)
#         self.emission = nn.Linear(output_dim, output_dim)
#         self.emission_action = nn.Linear(output_dim, output_action_dim)
#
#         if tanh:
#             self.particle_predictor = nn.Sequential(
#                 self.particle_predictor, nn.Tanh()
#             )
#
#     def forward(self, states, prev_hidden=None):
#         """
#         :param states: B x N x state_dim
#         :param pstep: 1 or 2
#         :return:
#         """
#         '''encode node'''
#         # obj_encode = self.obj_encoder(states)
#         obj_encode = states
#         # obj_encode = torch.cat([self.ini_conditions(obj_encode[:, :1]), obj_encode[:, 1:]], dim=1)
#         obj_encode = self.particle_predictor(obj_encode)
#         obj_encode = self.t_trans(obj_encode, prev_hidden)
#         obj_prediction = self.emission(obj_encode)
#         action_prediction = self.emission_action(obj_encode)
#
#         if self.output_action_dim is not None:
#             return obj_prediction, obj_encode, action_prediction
#         else:
#             return obj_prediction, obj_encode

# ======================================================================================================================

class SimplePropagationNetwork(nn.Module):

    def __init__(self, input_particle_dim, nf_particle, nf_effect, output_dim, output_action_dim = None,
                 tanh=False, residual=False, use_gpu=False, spectral_norm=False):

        super(SimplePropagationNetwork, self).__init__()

        self.use_gpu = use_gpu
        self.residual = residual
        self.output_action_dim = output_action_dim

        if spectral_norm:
            self.particle_predictor = SNParticlePredictor(input_particle_dim, nf_particle, output_dim)
        else:
            self.particle_predictor = ParticlePredictor(input_particle_dim, nf_particle, output_dim)
        # self.action_predictor = ParticlePredictor(input_particle_dim, nf_effect, output_action_dim)

        if tanh:
            self.particle_predictor = nn.Sequential(
                self.particle_predictor, nn.Tanh()
            )

    def forward(self, states, psteps):
        """
        :param states: B x N x state_dim
        :param pstep: 1 or 2
        :return:
        """
        '''encode node'''
        # obj_encode = self.obj_encoder(states)
        obj_encode = states
        obj_prediction = self.particle_predictor(obj_encode)
        # action_prediction = self.action_predictor(obj_encode)

        # if self.output_action_dim is not None:
        #     return obj_prediction, action_prediction
        # else:
        return obj_prediction

class KoopmanOperators(nn.Module, ABC):
    def __init__(self, state_dim, nf_particle, nf_effect, g_dim, u_dim, n_timesteps, n_blocks=1, residual=False):
        super(KoopmanOperators, self).__init__()

        self.residual = residual
        self.n_timesteps = n_timesteps
        self.u_dim = u_dim

        ''' state '''
        # TODO: state_dim * n_timesteps if not hankel. Pass hankel as parameter.
        input_particle_dim = state_dim * n_timesteps #+ g_dim #TODO: g_dim added for recursive sampling

        self.mapping = SimplePropagationNetwork(
            input_particle_dim=input_particle_dim, nf_particle=nf_particle,
            nf_effect=nf_effect, output_dim=g_dim, output_action_dim = u_dim, tanh=False,  # use tanh to enforce the shape of the code space
            residual=residual) # g * 2


        # self.linear_u_mapping = nn.Linear(g_dim * self.n_timesteps, u_dim * 2)
        # self.nlinear_u_mapping = nn.Sequential(nn.Linear(state_dim * n_timesteps, g_dim),
        #                                    nn.ReLU(),
        #                                    nn.Linear(state_dim, u_dim * 2))
        # self.nlinear_u_mapping = nn.Sequential(nn.Linear(state_dim * n_timesteps, state_dim),
        #                                   nn.ReLU(),
        #                                   nn.Linear(state_dim, u_dim + 1))

        # self.gru_u_mapping = nn.GRU(state_dim * n_timesteps, u_dim, num_layers = 1, batch_first=True)
        # self.nlinear_u_mapping = nn.Sequential(nn.Linear(state_dim * n_timesteps, state_dim),
        #                                   nn.ReLU(),
        #                                   nn.Linear(state_dim, u_dim))

        # the state for decoding phase is replaced with code of g_dim
        input_particle_dim = g_dim
        # print('state_decoder', 'node', input_particle_dim, 'edge', input_relation_dim)
        self.inv_mapping = SimplePropagationNetwork(
            input_particle_dim=input_particle_dim, nf_particle=nf_particle,
            nf_effect=nf_effect, output_dim=state_dim, tanh=False, residual=residual, spectral_norm=True)

        ''' dynamical system coefficient: A'''

        # self.system_identify = self.fit
        # self.simulate = self.rollout
        # self.step = self.linear_forward

        self.A_reg = torch.eye(g_dim // n_blocks).unsqueeze(0)
        # self.A = nn.Parameter(torch.ones(1, 1, n_blocks, g_dim // n_blocks, g_dim // n_blocks)
        #                       / (g_dim // n_blocks))
        # # self.B = nn.Parameter(torch.ones(1, 1, n_blocks, u_dim // n_blocks, g_dim // n_blocks)
        # #                       / (g_dim // n_blocks))
        # self.B = nn.Parameter(torch.ones(1, 1, n_blocks, u_dim // n_blocks,
        #                                  (g_dim * n_timesteps) // n_blocks)
        #                                 / ((g_dim * n_timesteps) // n_blocks))

        self.system_identify = self.fit_block_diagonal
        self.simulate = self.rollout_block_diagonal
        self.step = self.linear_forward_block_diagonal
        self.num_blocks = n_blocks

    def to_s(self, gcodes, psteps):
        """ state decoder """
        states = self.inv_mapping(states=gcodes, psteps=psteps)
        # regularize_state_Soft(states, rel_attrs, self.stat)
        return states

    def to_g(self, states, psteps):
        """ state encoder """
        return self.mapping(states=states, psteps=psteps)

    def to_u(self, states, temp, ignore=False):
        """ state encoder """
        # u_dist = self.nlinear_u_mapping(states.reshape(-1, states.shape[-1]))
        if not ignore:
            u_dist, _ = self.gru_u_mapping(states)
            u_dist = u_dist.reshape(*states.shape[:-1], -1)
            u = F.sigmoid(u_dist * temp)

            # Note: Binarize
            u_soft = u
            zeros = torch.zeros_like(u)
            ones = 1 - zeros
            u_hard = torch.where(u > 0.5, ones, zeros)
            u = u_hard - u_soft.detach() + u_soft

            # u_dist = F.relu(u_dist) # Note: Is relu necessary here?
            # u = F.gumbel_softmax(u_dist, tau=1/temp, hard=True)[..., 1:]
        else:
            u_dist = torch.zeros(*states.shape[:-1], self.u_dim).to(states.device)
            u = u_dist

        return u, u_dist

    # def to_g(self, states, hidden):
    #     """ state encoder """
    #     return self.mapping(states=states, prev_hidden=hidden)

    def fit_block_diagonal(self, G, H, U, I_factor):
        # TODO: it makes no sense that f_1(t+1) can contribute to f_2(t).
        #  but it makes sense the other way around.
        #  HOWEVER: We should simplify by assuming that
        #       all f_i(0) can interact with each other and all f_1(t) too.
        #       In any case, the regressor is in the null space of H,
        #       which I guess it is the same as we are doing now. Isn't it?
        #  A way to do it:
        #        G is the concatenation of time 0, 1, 2 and U too.
        #        Least squares will be done with H of size (f),
        #        G of size (3f) and u of size (3*usize)
        bs, T, N, D = G.size()

        a_dim = U.shape[-1]

        assert D % self.num_blocks == 0
        block_size = D // self.num_blocks
        a_block_size = a_dim // self.num_blocks

        G, H, U = G.reshape(bs, T, N, -1, block_size), \
                  H.reshape(bs, T, N, -1, block_size), \
                  U.reshape(bs, T, N, -1, a_block_size)
        G, H, U = G.permute(0, 3, 1, 2, 4), H.permute(0, 3, 1, 2, 4), U.permute(0, 3, 1, 2, 4)


        GU = torch.cat([G, U], -1)
        '''B x (D) x D'''
        AB = torch.bmm(
            self.batch_pinv(GU.reshape(bs * self.num_blocks, T * N, block_size + a_block_size), I_factor),
            H.reshape(bs * self.num_blocks, T * N, block_size)
        )
        A = AB[:, :block_size].reshape(bs, self.num_blocks, block_size, block_size)
        B = AB[:, block_size:].reshape(bs, self.num_blocks, a_block_size, block_size)
        fit_err = 0
        # fit_err = H.reshape(bs, T * N, D) - torch.bmm(GU, AB)
        # fit_err = torch.sqrt((fit_err ** 2).mean())

        # AB = AB[:, None].repeat(1, T, 1, 1).reshape(-1, block_size + a_block_size, block_size)
        # GU, H = GU.reshape(-1, 1, block_size + a_block_size), \
        #         H.reshape(-1, 1, block_size)
        # GUAB = torch.bmm(GU, AB)
        # fit_err = H - GUAB
        # fit_err = torch.sqrt((fit_err ** 2).sum(-1).mean())
        #
        #     # P = torch.bmm(
        #     #     self.batch_pinv(GUAB, I_factor),
        #     #     H
        #     # )
        #     # res_fit_err = H - torch.bmm(GUAB, P)
        #     # if res_fit_err.mean() > 0:
        #     #     print('it is different! by: ', res_fit_err.mean().item())

        reg_mask = torch.zeros_like(A)
        ids = torch.arange(0, reg_mask.shape[-1])
        reg_mask[..., ids, ids] = 0.01
        A_inv = torch.inverse(A + reg_mask).reshape(bs, self.num_blocks, block_size, block_size)
        A_inv = torch.clamp(A_inv, min = -1.5, max = 1.5)

        return A, B, A_inv, fit_err

    # def fit_block_diagonal(self, G, H, U, I_factor):
    #     # TODO: it makes no sense that f_1(t+1) can contribute to f_2(t).
    #     #  but it makes sense the other way around.
    #     #  HOWEVER: We should simplify by assuming that
    #     #       all f_i(0) can interact with each other and all f_1(t) too.
    #     #       In any case, the regressor is in the null space of H,
    #     #       which I guess it is the same as we are doing now. Isn't it?
    #     #  A way to do it:
    #     #        G is the concatenation of time 0, 1, 2 and U too.
    #     #        Least squares will be done with H of size (f),
    #     #        G of size (3f) and u of size (3*usize)
    #     bs, T, N, D = G.size()
    #
    #     a_dim = U.shape[-1]
    #
    #     assert D % self.num_blocks == 0
    #     block_size = D // self.num_blocks
    #     a_block_size = a_dim // self.num_blocks
    #
    #     H = H.reshape(*H.shape[:-1], -1, self.n_timesteps)[..., -1]
    #     G, H, U = G.reshape(bs, T, N, -1, block_size), \
    #               H.reshape(bs, T, N, -1, block_size // self.n_timesteps), \
    #               U.reshape(bs, T, N, -1, a_block_size)
    #     G, H, U = G.permute(0, 3, 1, 2, 4), H.permute(0, 3, 1, 2, 4), U.permute(0, 3, 1, 2, 4)
    #
    #     GU = torch.cat([G, U], -1)
    #     '''B x (D) x D'''
    #     AB = torch.bmm(
    #         self.batch_pinv(GU.reshape(bs * self.num_blocks, T * N, block_size + a_block_size), I_factor),
    #         H.reshape(bs * self.num_blocks, T * N, block_size // self.n_timesteps)
    #     )
    #
    #     A = AB[:, :block_size].reshape(bs, self.num_blocks, block_size, block_size // self.n_timesteps)
    #     B = AB[:, block_size:].reshape(bs, self.num_blocks, a_block_size, block_size // self.n_timesteps)
    #
    #
    #     AB = AB[:, None].repeat(1, T, 1, 1).reshape(-1, block_size + a_block_size, block_size // self.n_timesteps)
    #     GU, H = GU.reshape(-1, 1, block_size + a_block_size), \
    #             H.reshape(-1, 1, block_size // self.n_timesteps)
    #     GUAB = torch.bmm(GU, AB)
    #     fit_err = H - GUAB
    #     fit_err = torch.sqrt((fit_err ** 2).sum(-1).mean())
    #
    #     # P = torch.bmm(
    #     #     self.batch_pinv(GUAB, I_factor),
    #     #     H
    #     # )
    #     # res_fit_err = H - torch.bmm(GUAB, P)
    #     # if res_fit_err.mean() > 0:
    #     #     print('it is different! by: ', res_fit_err.mean().item())
    #
    #     # reg_mask = torch.zeros_like(A)
    #     # ids = torch.arange(0, reg_mask.shape[-1])
    #     # reg_mask[..., ids, ids] = 0.01
    #     # A_inv = torch.inverse(A + reg_mask).reshape(bs, self.num_blocks, block_size, block_size)
    #     # A_inv = torch.clamp(A_inv, min = -1.5, max = 1.5)
    #     A_inv = 0
    #
    #     return A, B, A_inv, fit_err

    def fit_with_A(self, G, H, U, I_factor):

        bs, T, N, D = G.size()
        a_dim = U.shape[-1]

        assert D % self.num_blocks == 0
        block_size = D // self.num_blocks
        a_block_size = a_dim // self.num_blocks

        A = self.A.repeat(bs, T, 1, 1, 1)
        H_prime = torch.bmm(G.reshape(-1, 1, block_size), A.reshape(-1, block_size, block_size  )).reshape(bs, T, N, D)
        res = H - H_prime
        res, U = res.reshape(bs, T, N, -1, block_size), \
                  U.reshape(bs, T, N, -1, a_block_size)
        res, U = res.permute(0, 3, 1, 2, 4), U.permute(0, 3, 1, 2, 4)

        U = U.reshape(bs, T, N, -1, a_block_size)
        U = U.permute(0, 3, 1, 2, 4)

        # GU = torch.cat([G, U], -1)
        '''B x (D) x D'''

        # res = H - H_prime
        B = torch.bmm(self.batch_pinv(U.reshape(bs * self.num_blocks, T * N, a_block_size), I_factor),
                      res.reshape(bs * self.num_blocks, T * N, block_size)
                      )

        B = B.reshape(bs, self.num_blocks, a_block_size, block_size)
        fit_err = 0
        # fit_err = H.reshape(bs, T * N, D) - torch.bmm(GU, AB)
        # fit_err = torch.sqrt((fit_err ** 2).mean())

        return A[:,0], B, fit_err

    def fit_with_B(self, G, H, U, I_factor):

        bs, T, N, D = G.size()
        a_dim = U.shape[-1]
        assert D % self.num_blocks == 0
        block_size = D // self.num_blocks
        a_block_size = a_dim // self.num_blocks

        B = self.B.repeat(bs, T, 1, 1, 1)

        action = torch.bmm(U.reshape(-1, 1, a_block_size), B.reshape(-1, a_block_size, block_size  )).reshape(bs, T, N, D)
        res = H - action
        res, G = res.reshape(bs, T, N, -1, block_size), \
                 G.reshape(bs, T, N, -1, block_size)
        res, G = res.permute(0, 3, 1, 2, 4), G.permute(0, 3, 1, 2, 4)

        # G = G.reshape(bs, T, N, -1, block_size)
        # G = G.permute(0, 3, 1, 2, 4)

        # GU = torch.cat([G, U], -1)
        '''B x (D) x D'''

        A = torch.bmm(self.batch_pinv(G.reshape(bs * self.num_blocks, T * N, block_size), I_factor),
                      res.reshape(bs * self.num_blocks, T * N, block_size)
                      )

        A = A.reshape(bs, self.num_blocks, block_size, block_size)
        fit_err = 0
        # fit_err = H.reshape(bs, T * N, D) - torch.bmm(GU, AB)
        # fit_err = torch.sqrt((fit_err ** 2).mean())

        return A, B[:,0], fit_err

    def fit_with_AB(self, bs):
        A = self.A.repeat(bs, 1, 1, 1, 1)
        B = self.B.repeat(bs, 1, 1, 1, 1)
        # fit_err = H.reshape(bs, T * N, D) - torch.bmm(GU, AB)
        # fit_err = torch.sqrt((fit_err ** 2).mean())

        return A[:,0], B[:,0]

    # def fit_block_diagonal_A(self, G, H, I_factor):
    #
    #     bs, T, N, D = G.size()
    #
    #     assert D % self.num_blocks == 0 and D % 2 == 0
    #     block_size = D // self.num_blocks
    #
    #     G, H = G.reshape(bs, T, N, -1, block_size), H.reshape(bs, T, N, -1, block_size)
    #     G, H = G.permute(0, 3, 1, 2, 4), H.permute(0, 3, 1, 2, 4)
    #
    #     '''B x (D) x D'''
    #     A = torch.bmm(
    #         self.batch_pinv(G.reshape(bs * self.num_blocks, T * N, block_size), I_factor),
    #         H.reshape(bs * self.num_blocks, T * N, block_size)
    #     )
    #     fit_err = None
    #     # A_pinv = None
    #     A_pinv = self.batch_pinv(A, I_factor).reshape(bs, self.num_blocks, block_size, block_size)
    #
    #     # reg_mask = torch.zeros_like(A)
    #     # ids = torch.arange(0, reg_mask.shape[-1])
    #     # reg_mask[..., ids, ids] = 0.05
    #     # A_pinv = torch.inverse(A + reg_mask).reshape(bs, self.num_blocks, block_size, block_size)
    #
    #     A = A.reshape(bs, self.num_blocks, block_size, block_size)
    #
    #     return A, A_pinv, fit_err

    def linear_forward_block_diagonal(self, g, u, A, B):
        """
        :param g: B x N x D
        :return:
        """
        ''' B x N x R D '''
        bs, dim = g.shape
        # dim = dim // self.n_timesteps #TODO:HANKEL This is for hankel view
        block_size = A.shape[-2]
        block_size_2 = A.shape[-1]
        a_block_size = B.shape[-2]
        aug_g, u = g.reshape(-1, 1, block_size), u.reshape(-1, 1, a_block_size)

        new_g = torch.bmm(aug_g, A.reshape(-1, block_size, block_size_2  )).reshape(bs, 1, dim) + \
                torch.bmm(u    , B.reshape(-1, a_block_size, block_size_2)).reshape(bs, 1, dim)
        return new_g

    def linear_forward_block_diagonal_no_input(self, g, A):
        """
        :param g: B x N x D
        :return:
        """
        ''' B x N x R D '''
        bs, dim = g.shape
        # dim = dim // self.n_timesteps #TODO:HANKEL This is for hankel view
        block_size = A.shape[-2]
        block_size_2 = A.shape[-1]
        aug_g = g.reshape(-1, 1, block_size)
        new_g = torch.bmm(aug_g, A.reshape(-1, block_size, block_size_2  )).reshape(bs, 1, dim)

        return new_g

    # def linear_forward_block_diagonal_no_input(self, g, A):
    #     """
    #     :param g: B x N x D
    #     :return:
    #     """
    #     ''' B x N x R D '''
    #     bs, dim = g.shape
    #     block_size = A.shape[-2]
    #     block_size_2 = A.shape[-1]
    #     aug_g= g.reshape(-1, 1, block_size)
    #
    #     new_g = torch.bmm(aug_g, A.reshape(-1, block_size, block_size_2  )).reshape(bs, 1, dim)
    #     return new_g

    # def rollout_block_diagonal(self, g, u, T, A, B):
    #     """
    #     :param g: B x N x D
    #     :param rel_attrs: B x N x N x R
    #     :param T:
    #     :return:
    #     """
    #     g_list = []
    #     if u is not None:
    #         for t in range(T):
    #             g = self.linear_forward_block_diagonal(g, u[:, t], A, B)[:, 0]  # For single object
    #             g_list.append(g[:, None, :])
    #     else:
    #         for t in range(T):
    #             g = self.linear_forward_block_diagonal_no_input(g, A)[:, 0]  # For single object
    #             g_list.append(g[:, None, :])
    #     return torch.cat(g_list, 1)

    # def rollout_block_diagonal(self, g, u, T, A, B):
    #     """
    #     :param g: B x N x D
    #     :param rel_attrs: B x N x N x R
    #     :param T:
    #     :return:
    #     """
    #     g_list = []
    #     if u is not None:
    #         for t in range(T):
    #             g_new = self.linear_forward_block_diagonal(g, u[:, t], A, B)[:, 0]  # For single object
    #             g = torch.cat([g.reshape(*g.shape[:-1], -1, self.n_timesteps)[..., 1:],
    #                            g_new[..., None]], dim=-1)
    #             g = g.reshape(*g.shape[:-2], -1)
    #             g_list.append(g_new[:, None, :])
    #     else:
    #         for t in range(T):
    #             g = self.linear_forward_block_diagonal_no_input(g, A)[:, 0]  # For single object
    #             g_list.append(g[:, None, :])
    #     return torch.cat(g_list, 1)

    # def rollout_block_diagonal(self, g, u, T, A, B):
    #     """
    #     :param g: B x N x D
    #     :param rel_attrs: B x N x N x R
    #     :param T:
    #     :return:
    #     """
    #     g_list = []
    #     if u is not None:
    #         for t in range(T):
    #             g_new = self.linear_forward_block_diagonal(g, u[:, t], A, B)[:, 0]  # For single object
    #             g = torch.cat([g.reshape(*g.shape[:-1], -1, self.n_timesteps)[..., 1:],
    #                            g_new[..., None]], dim=-1)
    #             g = g.reshape(*g.shape[:-2], -1)
    #             g_list.append(g_new[:, None, :])
    #     else:
    #         for t in range(T):
    #             g_new = self.linear_forward_block_diagonal_no_input(g, A)[:, 0]   # For single object
    #             g = torch.cat([g.reshape(*g.shape[:-1], -1, self.n_timesteps)[..., 1:],
    #                        g_new[..., None]], dim=-1)
    #             g = g.reshape(*g.shape[:-2], -1)
    #             g_list.append(g_new[:, None, :])
    #     return torch.cat(g_list, 1)

    def rollout_block_diagonal(self, g, u, T, A, B):
        """
        :param g: B x N x D
        :param rel_attrs: B x N x N x R
        :param T:
        :return:
        """
        g_list = []
        if u is not None:
            for t in range(T):
                g = self.linear_forward_block_diagonal(g, u[:, t], A, B)[:, 0]  # For single object
                g_list.append(g[:, None, :])
        else:
            for t in range(T):
                g = self.linear_forward_block_diagonal_no_input(g, A)[:, 0]  # For single object
                g_list.append(g[:, None, :])
        return torch.cat(g_list, 1)

    @staticmethod
    def batch_pinv(x, I_factor):
        """
        :param x: B x N x D (N > D)
        :param I_factor:
        :return:
        """

        B, N, D = x.size()

        if N < D:
            x = torch.transpose(x, 1, 2)
            N, D = D, N
            trans = True
        else:
            trans = False

        x_t = torch.transpose(x, 1, 2)

        use_gpu = torch.cuda.is_available()
        I = torch.eye(D)[None, :, :].repeat(B, 1, 1)
        if use_gpu:
            I = I.to(x.device)

        x_pinv = torch.bmm(
            torch.inverse(torch.bmm(x_t, x) + I_factor * I),
            x_t
        )

        if trans:
            x_pinv = torch.transpose(x_pinv, 1, 2)

        return x_pinv

    # @staticmethod
    # def get_aug(G, rel_attrs):
    #     """
    #     :param G: B x T x N x D
    #     :param rel_attrs:  B x N x N x R
    #     :return:
    #     """
    #     B, T, N, D = G.size()
    #     R = rel_attrs.size(-1)
    #
    #     sumG_list = []
    #     for i in range(R):
    #         ''' B x T x N x N '''
    #         adj = rel_attrs[:, :, :, i][:, None, :, :].repeat(1, T, 1, 1)
    #         sumG = torch.bmm(
    #             adj.reshape(B * T, N, N),
    #             G.reshape(B * T, N, D)
    #         ).reshape(B, T, N, D)
    #         sumG_list.append(sumG)
    #
    #     augG = torch.cat(sumG_list, 3)
    #
    #     return augG

