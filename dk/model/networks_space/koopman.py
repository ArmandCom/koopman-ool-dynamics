from abc import ABC

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from model.networks_space.spectral_norm import SpectralNorm
from utils import tracker_util as tut
from utils import util as ut


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
        self.linear_1 = nn.Linear(hidden_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: [n_particles, input_size]
        Returns:
            [n_particles, output_size]
        """
        x = self.relu(self.linear_0(x))
        x = self.relu(self.linear_1(x))

        return self.linear_2(x)

class SNParticlePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SNParticlePredictor, self).__init__()

        self.linear_0 = SpectralNorm(nn.Linear(input_size, hidden_size))
        self.linear_1 = SpectralNorm(nn.Linear(hidden_size, hidden_size))
        self.linear_2 = SpectralNorm(nn.Linear(hidden_size, output_size))
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: [n_particles, input_size]
        Returns:
            [n_particles, output_size]
        """
        x = self.relu(self.linear_0(x))
        x = self.relu(self.linear_1(x))

        return self.linear_2(x)

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
    def __init__(self, state_dim, nf_particle, nf_effect, g_dim, u_dim, n_timesteps, n_blocks=1, residual=False, deriv_in_state=False, fixed_A=False, fixed_B=False, num_sys=0):
        super(KoopmanOperators, self).__init__()

        self.residual = residual
        self.n_timesteps = n_timesteps
        self.u_dim = u_dim

        if deriv_in_state and n_timesteps > 2:
            first_deriv_dim = n_timesteps - 1
            sec_deriv_dim = n_timesteps - 2
        else:
            first_deriv_dim = 0
            sec_deriv_dim = 0

        ''' state '''
        # TODO: state_dim * n_timesteps if not hankel. Pass hankel as parameter.
        input_particle_dim = state_dim * (n_timesteps + first_deriv_dim + sec_deriv_dim) #+ g_dim #TODO: g_dim added for recursive sampling

        self.mapping = SimplePropagationNetwork(
            input_particle_dim=input_particle_dim, nf_particle=nf_particle,
            nf_effect=nf_effect, output_dim=g_dim, output_action_dim = u_dim, tanh=False,  # use tanh to enforce the shape of the code space
            residual=residual) # g * 2

        self.composed_mapping = SimplePropagationNetwork(
            input_particle_dim=input_particle_dim, nf_particle=nf_particle,
            nf_effect=nf_effect, output_dim=2 * g_dim, output_action_dim = u_dim, tanh=False,  # use tanh to enforce the shape of the code space
            residual=residual) # g * 2


        self.gru_u_mapping = nn.GRU(input_particle_dim, u_dim, num_layers = 1, batch_first=True)
        self.linear_u_mapping = nn.Linear(u_dim, u_dim * 2)
        #
        self.nlinear_u_mapping = nn.Sequential(nn.Linear(input_particle_dim, state_dim),
                                          nn.ReLU(),
                                          nn.Linear(state_dim, u_dim * 2))

        # self.nlinear_u_mapping = nn.Sequential(nn.Linear(g_dim, state_dim),
        #                                        nn.ReLU(),
        #                                        nn.Linear(state_dim, u_dim * 2))

        # the state for decoding phase is replaced with code of g_dim
        input_particle_dim = g_dim
        # print('state_decoder', 'node', input_particle_dim, 'edge', input_relation_dim)
        self.inv_mapping = SimplePropagationNetwork(
            input_particle_dim=input_particle_dim, nf_particle=nf_particle,
            nf_effect=nf_effect, output_dim=state_dim, tanh=False, residual=residual, spectral_norm=True) # TRUE

        ''' dynamical system coefficient: A'''

        # self.system_identify = self.fit
        # self.simulate = self.rollout
        # self.step = self.linear_forward

        self.A_reg = torch.eye(g_dim // n_blocks).unsqueeze(0)

        if fixed_A:
            # self.A = nn.Parameter( torch.randn((1, n_blocks, g_dim // n_blocks, g_dim // n_blocks), requires_grad=True) * .1 + (1 / (g_dim // n_blocks)))
            self.A = nn.Parameter(torch.zeros(1, n_blocks, g_dim // n_blocks, g_dim // n_blocks, requires_grad=True) + torch.eye(g_dim // n_blocks)[None, None])
        if fixed_B:
            self.B = nn.Parameter( torch.randn((1, n_blocks, u_dim // n_blocks, g_dim // n_blocks), requires_grad=True) * .1 + (1 / (g_dim // n_blocks)))
        if num_sys > 0:
            # self.A = nn.Parameter( torch.randn((1, num_sys, g_dim // n_blocks, g_dim // n_blocks), requires_grad=True) * .1 + (1 / (g_dim // n_blocks)))
            self.A = nn.Parameter(torch.zeros(1, num_sys, g_dim // n_blocks, g_dim // n_blocks, requires_grad=True) + torch.eye(g_dim // n_blocks)[None, None])
            # ids = torch.arange(0,self.A.shape[-1])
            # self.A[..., ids, ids] = 1
            # self.A = nn.Parameter(self.A)
            self.selector_fc = nn.Sequential(nn.Linear(g_dim, g_dim),
                                              nn.ReLU(),
                                              nn.Linear(g_dim, num_sys),
                                            nn.ReLU())


        self.softmax = nn.Softmax(dim=-1)
        self.st_gumbel_softmax = tut.STGumbelSoftmax(-1)
        self.round = tut.Round()
        self.st_gumbel_sigmoid = tut.STGumbelSigmoid()

        self.system_identify = self.fit_block_diagonal
        self.system_identify_with_A = self.fit_with_A
        self.system_identify_with_compositional_A = self.fit_with_compositional_A
        # self.system_identify = self.fit_across_objects
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

    def to_selector(self, g, temp=1):
        """ state encoder """
        obs_shape = g.shape
        sel_logit = self.selector_fc(g.reshape(-1, obs_shape[-1]))
        sel_sm = self.softmax(sel_logit)
        tut.norm_grad(sel_sm, 2)
        selector = self.st_gumbel_softmax(sel_sm)
        return selector

    def to_composed_g(self, states, psteps, temp=1):
        """ state encoder """
        # TODO: encode with differences
        g, g_mask_logit = self.composed_mapping(states=states, psteps=psteps).chunk(2, dim=-1)

        g_mask_logit = g_mask_logit.tanh() * 8.8
        g_mask_post = ut.NumericalRelaxedBernoulli(logits=g_mask_logit, temperature=temp)
        # Unbounded
        g_mask_sample = g_mask_post.rsample()
        g_mask = self.round(torch.sigmoid(g_mask_sample))
        return g, g_mask, g_mask_logit

    def to_u(self, states, temp, ignore=False):
        # TODO: Encode differences to minimize increments.
        """ state encoder """
        # u_dist = self.nlinear_u_mapping(states.reshape(-1, states.shape[-1]))
        # if not ignore:
        #     u_logit, _ = self.gru_u_mapping(states)
        #     u_logit = u_logit.reshape(*states.shape[:-1], -1)
        #     # u = F.sigmoid(u_dist * temp)
        #     u_post = u_logit.sigmoid()
        #     u = self.st_gumbel_sigmoid(u_post)
        #
        #     # Note: Binarize
        #     # u_soft = u
        #     # zeros = torch.zeros_like(u)
        #     # ones = 1 - zeros
        #     # u_hard = torch.where(u > 0.5, ones, zeros)
        #     # u = u_hard - u_soft.detach() + u_soft
        #
        #     # u_dist = F.relu(u_dist) # Note: Is relu necessary here?
        #     # u = F.gumbel_softmax(u_dist, tau=1/temp, hard=True)[..., 1:]
        # else:
        #     u_logit = torch.zeros(*states.shape[:-1], self.u_dim).to(states.device)
        #     u = u_logit
        #     u_post = u_logit

        if not ignore:
            # Option 2: Absolute inputs
            # rnn_out, _ = self.gru_u_mapping(states)
            # rnn_out_shape = rnn_out.shape
            # u_logit, u_feat = self.linear_u_mapping(rnn_out.reshape(-1, rnn_out_shape[-1])).reshape(*rnn_out_shape[:2], -1).chunk(2, dim=-1)
            # # TODO: 0s the first and last
            # u_logit = u_logit.reshape(*states.shape[:-1], -1).tanh() * 8.8
            # u_post = ut.NumericalRelaxedBernoulli(logits=u_logit, temperature=temp)
            # # Unbounded
            # u_y = u_post.rsample()
            # # in (0, 1)
            # u = self.round(torch.sigmoid(u_y))
            # u = u * u_feat

            # Option 3: Difference with non-linear mapping
            states = states[:, 1:]
            states_shape = states.shape
            # rnn_out, _ = self.gru_u_mapping(states)
            # rnn_out_shape = rnn_out.shape
            # u_logit, u_feat = self.linear_u_mapping(rnn_out.reshape(-1, rnn_out_shape[-1])).reshape(*rnn_out_shape[:2], -1).chunk(2, dim=-1)
            u_logit, u_feat = self.nlinear_u_mapping(states.reshape(-1, states_shape[-1])).reshape(*states_shape[:2], -1).chunk(2, dim=-1)
            # TODO: Try without u_feat
            u_logit = u_logit.tanh() * 8.8
            u_post = ut.NumericalRelaxedBernoulli(logits=u_logit, temperature=temp)
            # Unbounded
            u_y = u_post.rsample()
            # in (0, 1)
            u = self.round(torch.sigmoid(u_y))

            # u = self.sequential_switching(torch.zeros_like(u[:, :1]), u)
            # u = self.accumulate_obs(torch.zeros_like(u[:, :1]), u)
            u = torch.cat([torch.zeros_like(u[:, :1]), u], dim =1)
            # u = u * u_feat

            # Option 4: u deterministic
            # states = states[:, 1:]
            # states_shape = states.shape
            # # rnn_out, _ = self.gru_u_mapping(states)
            # # rnn_out_shape = rnn_out.shape
            # # u_logit, u_feat = self.linear_u_mapping(rnn_out.reshape(-1, rnn_out_shape[-1])).reshape(*rnn_out_shape[:2], -1).chunk(2, dim=-1)
            # u_logit, u_feat = self.nlinear_u_mapping(states.reshape(-1, states_shape[-1])).reshape(*states_shape[:2], -1).chunk(2, dim=-1)
            # u = self.round(torch.sigmoid(u_logit))
            # u = torch.cat([torch.zeros_like(u[:, :1]), u], dim =1)
            # u_post = None
            # u_logit = None

        else:
            u = torch.zeros(*states.shape[:-1], self.u_dim).to(states.device)
            u_post = None
            u_logit = None
        return u, u_post, u_logit

    # def to_g(self, states, hidden):
    #     """ state encoder """
    #     return self.mapping(states=states, prev_hidden=hidden)

    def fit_across_objects(self, G, H, U, I_factor, n_objects=1):

        bsO, T, N, D = G.size()
        a_dim = U.shape[-1]

        G = G.reshape(bsO // n_objects, n_objects, T, N, D)
        U = U.reshape(bsO // n_objects, n_objects, T, N, a_dim)
        bs = bsO//n_objects

        assert D % self.num_blocks == 0
        block_size = D // self.num_blocks
        a_block_size = a_dim // self.num_blocks

        G, H, U = G.reshape(bs, n_objects * T, N, -1, block_size), \
                  H.reshape(bs, n_objects * T, N, -1, block_size), \
                  U.reshape(bs, n_objects * T, N, -1, a_block_size)
        G, H, U = G.permute(0, 3, 1, 2, 4), H.permute(0, 3, 1, 2, 4), U.permute(0, 3, 1, 2, 4)

        GU = torch.cat([G, U], -1)
        '''B x (D) x D'''
        AB = torch.bmm(
            self.batch_pinv(GU.reshape(bs * self.num_blocks, n_objects * T * N, block_size + a_block_size), I_factor),
            H.reshape(bs * self.num_blocks, n_objects * T * N, block_size)
        )
        AB = AB[:, None].repeat_interleave(n_objects, dim=1).reshape(bsO * self.num_blocks, *AB.shape[1:])

        A = AB[:, :block_size].reshape(bsO, self.num_blocks, block_size, block_size)
        B = AB[:, block_size:].reshape(bsO, self.num_blocks, a_block_size, block_size)
        fit_err = 0

        # reg_mask = torch.zeros_like(A)
        # ids = torch.arange(0, reg_mask.shape[-1])
        # reg_mask[..., ids, ids] = 0.01
        # A_inv = torch.inverse(A + reg_mask).reshape(bsO, self.num_blocks, block_size, block_size)
        # A_inv = torch.clamp(A_inv, min = -1.5, max = 1.5)
        A_inv = None

        return A, B, A_inv, fit_err

    def fit_block_diagonal(self, G, H, U, I_factor, n_objects=1):
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

        # reg_mask = torch.zeros_like(A)
        # ids = torch.arange(0, reg_mask.shape[-1])
        # reg_mask[..., ids, ids] = 0.01
        # A_inv = torch.inverse(A + reg_mask).reshape(bs, self.num_blocks, block_size, block_size)
        # A_inv = torch.clamp(A_inv, min = -1.5, max = 1.5)
        A_inv = 0

        return A, B, A_inv, fit_err

    # def fit_with_A(self, G, H, U, I_factor, n_objects=1):
    #
    #     bs, T, N, D = G.size()
    #     a_dim = U.shape[-1]
    #
    #     assert D % self.num_blocks == 0
    #     block_size = D // self.num_blocks
    #     a_block_size = a_dim // self.num_blocks
    #
    #     G, H, U = G.reshape(bs, T, N, -1, block_size), \
    #               H.reshape(bs, T, N, -1, block_size), \
    #               U.reshape(bs, T, N, -1, a_block_size)
    #     G, H, U = G.permute(0, 3, 1, 2, 4), H.permute(0, 3, 1, 2, 4), U.permute(0, 3, 1, 2, 4)
    #
    #
    #     GU = torch.cat([G, U], -1)
    #     '''B x (D) x D'''
    #     AB = torch.bmm(
    #         self.batch_pinv(GU.reshape(bs * self.num_blocks, T * N, block_size + a_block_size), I_factor),
    #         H.reshape(bs * self.num_blocks, T * N, block_size)
    #     )
    #     A = AB[:, :block_size].reshape(bs, self.num_blocks, block_size, block_size)
    #     B = AB[:, block_size:].reshape(bs, self.num_blocks, a_block_size, block_size)
    #     fit_err = 0
    #     A_inv = 0
    #
    #     return A, B, A_inv, fit_err

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

        A = self.A.repeat(bs * T, 1, 1, 1)
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

        # fit_err = H.reshape(bs, T * N, D) - torch.bmm(GU, AB)
        # fit_err = torch.sqrt((fit_err ** 2).mean())

        return A[:bs], B

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

    def fit_with_compositional_A(self, G, H, U, I_factor, with_B = True):

        bs, T, N, D = G.size()
        a_dim = U.shape[-1]

        selector = self.to_selector(G)

        # TODO: This fit can also be done by OLS, replicating G by num_sys, and therefore fitting for each selected system per observation.
        assert D % self.num_blocks == 0
        block_size = D // self.num_blocks
        a_block_size = a_dim // self.num_blocks

        S = selector # bs*T, sel_size
        A = torch.einsum('blij,bl->bij', self.A.repeat(bs * T, 1, 1, 1), S) # bs*T, A_dim, A_dim --> different at each batch item and timestep
        # print((A[0] - self.A[0,0]).mean(), (A[0] - self.A[0,1]).mean(), (A[0] - self.A[0,2]).mean(), S[0])
        if with_B:
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
        else:
            B = torch.zeros(bs, self.num_blocks, a_block_size, block_size).to(self.A.device)

        # fit_err = H.reshape(bs, T * N, D) - torch.bmm(GU, AB)
        # fit_err = torch.sqrt((fit_err ** 2).mean())

        return A[:bs], B, selector.reshape(bs, T, -1).transpose(-2, -1)

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

    def rollout_block_diagonal(self, g, u, T, A, B, clip=30):
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
        return torch.clamp(torch.cat(g_list, 1), min=-clip, max=clip)

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

    def accumulate_obs(self, obs_ini, increments):
        T = increments.shape[1]
        Tini = obs_ini.shape[1]
        assert Tini == 1

        obs = [obs_ini[:, 0]]
        for t in range(T):
            obs.append(obs[t] + increments[:, t])
        obs = torch.stack(obs, dim=1)
        return obs

    def sequential_switching(self, obs_ini, switches):
        '''

        Args:
            obs_ini: First value. By default should be 0
            all_obs: All u switches. {0,1}
            First value will compute the second value of the output

        Returns:

        '''
        T = switches.shape[1]
        Tini = obs_ini.shape[1]
        assert Tini == 1

        obs = [obs_ini[:, 0]]
        for t in range(T):
            obs.append(self.xor(obs[t], switches[:, t]))
        obs = torch.stack(obs, dim=1)
        return obs

    def xor(self, x, y):
        # print(((1-x)*y + (1-y)*x).shape, x.shape, y.shape)
        return (1-x)*y + (1-y)*x
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

