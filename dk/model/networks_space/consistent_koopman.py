from abc import ABC

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from model.networks_space.spectral_norm import SpectralNorm
from utils import tracker_util as tut
from utils import util as ut

def gaussian_init_(n_units, std=1):
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std/n_units]))
    Omega = sampler.sample((n_units, n_units))[..., 0]
    return Omega

class encoderNet(nn.Module):
    def __init__(self, N, b, ALPHA = 1):
        super(encoderNet, self).__init__()
        self.N = N
        self.tanh = nn.Tanh()

        self.fc1 = nn.Linear(self.N, 16*ALPHA)
        self.fc2 = nn.Linear(16*ALPHA, 16*ALPHA)
        self.fc3 = nn.Linear(16*ALPHA, b)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = x.view(-1, 1, self.N)
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.fc3(x)

        return x


class decoderNet(nn.Module):
    def __init__(self, N, b, ALPHA = 1):
        super(decoderNet, self).__init__()

        self.N = N
        self.b = b

        self.tanh = nn.Tanh()

        # TODO: Spectralnorm.
        self.fc1 = nn.Linear(b, 16*ALPHA)
        self.fc2 = nn.Linear(16*ALPHA, 16*ALPHA)
        self.fc3 = nn.Linear(16*ALPHA, N)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = x.view(-1, 1, self.b)
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.tanh(self.fc3(x))
        x = x.view(-1, 1, self.N)
        return x

class dynamics(nn.Module):
    def __init__(self, b, init_scale):
        super(dynamics, self).__init__()
        self.dynamics = nn.Linear(b, b, bias=False)
        self.dynamics.weight.data = gaussian_init_(b, std=1)
        U, _, V = torch.svd(self.dynamics.weight.data)
        self.dynamics.weight.data = torch.mm(U, V.t()) * init_scale
        # TODO: Recursively keep only k singular values.
        # TODO: if OLS G = g[:-1], H= g[1:], svd(g). --> Might not be right because you have to invert G and not H

    def forward(self, x):
        x = self.dynamics(x)
        return x


class dynamics_back(nn.Module):
    def __init__(self, b, omega):
        super(dynamics_back, self).__init__()
        self.dynamics = nn.Linear(b, b, bias=False)
        self.dynamics.weight.data = torch.pinverse(omega.dynamics.weight.data.t())

    def forward(self, x):
        x = self.dynamics(x)
        return x

class KoopmanOperators(nn.Module, ABC):
    # def __init__(self, state_dim, nf_particle, nf_effect, g_dim, u_dim, n_timesteps, n_blocks=1, residual=False, deriv_in_state=False, fixed_A=False, fixed_B=False, num_sys=0):
    def __init__(self, state_dim, nf_particle, nf_effect, g_dim, u_dim, n_timesteps, deriv_in_state=False, num_sys=0, alpha=1, init_scale=1):

        super(KoopmanOperators, self).__init__()

        self.n_timesteps = n_timesteps
        self.u_dim = u_dim

        if deriv_in_state and n_timesteps > 2:
            first_deriv_dim = n_timesteps - 1
            sec_deriv_dim = n_timesteps - 2
        else:
            first_deriv_dim = 0
            sec_deriv_dim = 0

        ''' state '''
        input_particle_dim = state_dim * (n_timesteps + first_deriv_dim + sec_deriv_dim) #+ g_dim # g_dim added for recursive sampling

        self.mapping = encoderNet(input_particle_dim, g_dim, ALPHA=alpha)
        self.composed_mapping = encoderNet(input_particle_dim, g_dim * 2, ALPHA=alpha)
        self.inv_mapping = decoderNet(state_dim, g_dim, ALPHA=alpha)
        self.dynamics = dynamics(g_dim, init_scale)
        self.backdynamics = dynamics_back(g_dim, self.dynamics)

        # self.gru_u_mapping = nn.GRU(input_particle_dim, u_dim, num_layers = 1, batch_first=True)
        # self.linear_u_mapping = nn.Linear(u_dim, u_dim * 2)
        #
        self.nlinear_u_mapping = nn.Sequential(nn.Linear(input_particle_dim, state_dim),
                                          nn.ReLU(),
                                          nn.Linear(state_dim, u_dim * 2))


        # if num_sys > 0:
        #     # self.A = nn.Parameter( torch.randn((1, num_sys, g_dim // n_blocks, g_dim // n_blocks), requires_grad=True) * .1 + (1 / (g_dim // n_blocks) **2))
        #     self.A = torch.zeros(1, num_sys, g_dim, g_dim)
        #     ids = torch.arange(0,self.A.shape[-1])
        #     self.A[..., ids, ids] = 1
        #     self.A = nn.Parameter(self.A, requires_grad=True)
        #     self.selector_fc = nn.Sequential(nn.Linear(g_dim, g_dim),
        #                                       nn.ReLU(),
        #                                       nn.Linear(g_dim, num_sys),
        #                                     nn.ReLU())


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

    def forward(self, x, mode='forward'):
        out = []
        out_back = []
        z = self.encoder(x.contiguous())
        q = z.contiguous()

        steps = x.shape[1]
        steps_back = x.shape[1]

        if mode == 'forward':
            for _ in range(steps):
                q = self.dynamics(q)
                out.append(self.decoder(q))

            out.append(self.decoder(z.contiguous()))
            return out, out_back

        if mode == 'backward':
            for _ in range(steps_back):
                q = self.backdynamics(q)
                out_back.append(self.decoder(q))

            out_back.append(self.decoder(z.contiguous()))
            return out, out_back

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

    def fit_block_diagonal(self, G, H, U, I_factor, n_objects=1):
        bs, T, N, D = G.size()
        a_dim = U.shape[-1]

        block_size = D
        a_block_size = a_dim

        G, H, U = G.reshape(bs, T, N, -1, block_size), \
                  H.reshape(bs, T, N, -1, block_size), \
                  U.reshape(bs, T, N, -1, a_block_size)
        G, H, U = G.permute(0, 3, 1, 2, 4), H.permute(0, 3, 1, 2, 4), U.permute(0, 3, 1, 2, 4)


        GU = torch.cat([G, U], -1)
        '''B x (D) x D'''
        AB = torch.bmm(
            self.batch_pinv(GU.reshape(bs, T * N, block_size + a_block_size), I_factor),
            H.reshape(bs, T * N, block_size)
        )
        A = AB[:, :block_size].reshape(bs, 1, block_size, block_size)
        B = AB[:, block_size:].reshape(bs, 1, a_block_size, block_size)
        fit_err = 0
        A_inv = 0

        return A, B, A_inv, fit_err

    def fit_with_A(self, G, H, U, I_factor):

        bs, T, N, D = G.size()
        a_dim = U.shape[-1]

        block_size = D
        a_block_size = a_dim

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
        B = torch.bmm(self.batch_pinv(U.reshape(bs, T * N, a_block_size), I_factor),
                      res.reshape(bs, T * N, block_size)
                      )

        B = B.reshape(bs, 1, a_block_size, block_size)

        # fit_err = H.reshape(bs, T * N, D) - torch.bmm(GU, AB)
        # fit_err = torch.sqrt((fit_err ** 2).mean())

        return A[:bs], B

    def fit_with_AB(self, bs):
        A = self.A.repeat(bs, 1, 1, 1, 1)
        B = self.B.repeat(bs, 1, 1, 1, 1)

        return A[:,0], B[:,0]

    def fit_with_compositional_A(self, G, H, U, I_factor, with_B = True):

        bs, T, N, D = G.size()
        a_dim = U.shape[-1]

        selector = self.to_selector(G)

        # TODO: This fit can also be done by OLS, replicating G by num_sys, and therefore fitting for each selected system per observation.
        block_size = D
        a_block_size = a_dim

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

            # res = H - H_prime
            B = torch.bmm(self.batch_pinv(U.reshape(bs * self.num_blocks, T * N, a_block_size), I_factor),
                          res.reshape(bs * self.num_blocks, T * N, block_size)
                          )

            B = B.reshape(bs, self.num_blocks, a_block_size, block_size)
        else:
            B = torch.zeros(bs, self.num_blocks, a_block_size, block_size).to(self.A.device)

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
        return (1-x)*y + (1-y)*x

