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

def gaussian_init_2dim(units, std=1):
    assert len(units) == 2
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std/units[0]]))
    Omega = sampler.sample((units[0], units[1]))[..., 0]
    return Omega

class encoderNet(nn.Module):
    def __init__(self, N, b, ALPHA = 1):
        super(encoderNet, self).__init__()
        self.N = N
        self.tanh = nn.Tanh()
        self.b = b

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
        x = self.fc3(x).view(-1, self.b)

        return x


class decoderNet(nn.Module):
    def __init__(self, N, b, ALPHA = 1, SN=False):
        super(decoderNet, self).__init__()

        self.N = N
        self.b = b

        self.tanh = nn.Tanh()

        if SN:
            self.fc1 = SpectralNorm(nn.Linear(b, 16*ALPHA))
            self.fc2 = SpectralNorm(nn.Linear(16*ALPHA, 16*ALPHA))
            self.fc3 = SpectralNorm(nn.Linear(16*ALPHA, N))
        else:
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
        x = x.view(-1, self.N)
        return x

class dynamics(nn.Module):
    def __init__(self, b, init_scale):
        super(dynamics, self).__init__()
        self.dynamics = nn.Linear(b, b, bias=False)

        #Option 1

        self.dynamics.weight.data = torch.zeros_like(self.dynamics.weight.data) + 0.001

        #Option 2

        # self.dynamics.weight.data = gaussian_init_(b, std=1)
        # U, S, V = torch.svd(self.dynamics.weight.data)
        # self.dynamics.weight.data = torch.mm(U, V.t()) * init_scale

        #Option 3

        # k = 5
        # S = torch.ones_like(S)
        # S[..., k:] = 0
        # self.dynamics.weight.data = torch.matmul(torch.matmul(U, torch.diag_embed(S)), V.t()) * init_scale
        # TODO: Recursively keep only k singular values.
        # TODO: if OLS G = g[:-1], H= g[1:], svd(g). --> Might not be right because you have to invert G and not H

    def forward(self, x):
        x = self.dynamics(x)
        return x

class u_dynamics(nn.Module):
    # TODO: clip grads after Sigmoid()
    def __init__(self, b, s, ud):
        super(u_dynamics, self).__init__()
        self.u_dynamics = nn.Linear(ud*b, b, bias=False)
        self.linear_u_mapping = nn.Linear(s, ud)
        self.linear_u_mapping.weight.data = gaussian_init_2dim([s, ud], std=1)
        self.nlinear_u_mapping = nn.Sequential(nn.Linear(s, s), nn.CELU(), self.linear_u_mapping)


        # self.u_dynamics.weight.data = gaussian_init_2dim([b, ud], std=1)
        self.u_dynamics.weight.data = torch.zeros_like(self.u_dynamics.weight.data) + 0.001
        # self.linear_u_mapping.weight.data = gaussian_init_2dim([b, ud], std=1)

        self.round = tut.Round()
        self.count = 0

    def forward(self, x, u=None, noise=True, propagate=True, temp=1):
        # x = x[:, 1:]

        if propagate:
            u = u[..., :, None]
            x = x[..., None, :]
            input = x*u
            out = self.u_dynamics(input.reshape(x.shape[0], -1))
            return out

        else:
            u_logit = self.nlinear_u_mapping(x)
            if noise and self.count<8000:
                u_logit = u_logit + torch.randn(u_logit.shape).to(u_logit.device)*0.1
                self.count+=1

            # Option 1: Deterministic
            # u_y = u_logit

            # Option 1: Stochastic
            u_logit = u_logit.tanh() * 8.8
            u_post = ut.NumericalRelaxedBernoulli(logits=u_logit, temperature=temp)
            # Unbounded
            u_y = u_post.rsample()

            u = self.round(torch.sigmoid(u_y / temp))
            # u = self.sequential_switching(torch.zeros_like(u[:, :1]), u)
            # u = self.accumulate_obs(torch.zeros_like(u[:, :1]), u)
            return u, u_logit

class dynamics_back(nn.Module):
    def __init__(self, b, omega):
        super(dynamics_back, self).__init__()
        self.dynamics_back = nn.Linear(b, b, bias=False)
        self.dynamics_back.weight.data = torch.pinverse(omega.dynamics.weight.data)
        # print(torch.norm(self.dynamics_back.weight.data  ),torch.norm(self.dynamics_back.weight.data - torch.pinverse(omega.dynamics.weight.data)))

    def forward(self, x):
        x = self.dynamics_back(x)
        return x

class KoopmanOperators(nn.Module, ABC):
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
        # self.composed_mapping = encoderNet(input_particle_dim, g_dim * 2, ALPHA=alpha)
        self.inv_mapping = decoderNet(state_dim, g_dim, ALPHA=alpha, SN=False)
        self.dynamics = dynamics(g_dim, init_scale)
        self.backdynamics = dynamics_back(g_dim, self.dynamics)
        self.u_dynamics = u_dynamics(g_dim, state_dim, u_dim)

        # self.gru_u_mapping = nn.GRU(input_particle_dim, u_dim, num_layers = 1, batch_first=True)
        # self.linear_u_mapping = nn.Linear(u_dim, u_dim * 2)
        #
        # self.nlinear_u_mapping = nn.Sequential(nn.Linear(input_particle_dim, state_dim),
        #                                   nn.ReLU(),
        #                                   nn.Linear(state_dim, u_dim * 2))


        self.softmax = nn.Softmax(dim=-1)
        self.st_gumbel_softmax = tut.STGumbelSoftmax(-1)
        self.round = tut.Round()
        self.st_gumbel_sigmoid = tut.STGumbelSigmoid()

        self.system_identify = self.fit_block_diagonal
        # self.system_identify_with_A = self.fit_with_A
        # self.system_identify_with_compositional_A = self.fit_with_compositional_A
        # self.system_identify = self.fit_across_objects
        self.simulate = self.rollout
        self.simulate_no_input = self.rollout_no_input
        # self.step = self.linear_forward_block_diagonal

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
        states = self.inv_mapping(gcodes)
        return states

    def to_g(self, states, psteps):
        """ state encoder """
        return self.mapping(states)

    def to_selector(self, g, temp=1):
        """ state encoder """
        obs_shape = g.shape
        sel_logit = self.selector_fc(g.reshape(-1, obs_shape[-1]))
        sel_sm = self.softmax(sel_logit)
        tut.norm_grad(sel_sm, 2)
        selector = self.st_gumbel_softmax(sel_sm)
        return selector

    def get_A_Ainv(self, get_A_inv=False, get_B=True):
        A = self.dynamics.dynamics.weight.data
        A_inv = None
        AA_inv = None
        B = None
        if get_A_inv:
            A_inv = self.backdynamics.dynamics_back.weight.data
            AA_inv = torch.mm(A, A_inv)
        if get_B:
            B = self.u_dynamics.u_dynamics.weight.data
        return A, A_inv, AA_inv, B

    def limit_rank_k(self, k=None):
        if k is not None:
            U, S, V = torch.svd(self.dynamics.dynamics.weight.data)
            print('Singular Values FW: ',S)
            S[..., k:] = 0 # Review if this is right
            self.dynamics.dynamics.weight.data = torch.matmul(torch.matmul(U, torch.diag_embed(S)), V.transpose(-2, -1))
            #
            U, S, V = torch.svd(self.backdynamics.dynamics_back.weight.data)
            S[..., k:] = 0
            print('Singular Values BW: ',S)
            self.backdynamics.dynamics_back.weight.data = torch.matmul(torch.matmul(U, torch.diag_embed(S)), V.transpose(-2, -1))

    def print_SV(self):
            U, S, V = torch.svd(self.dynamics.dynamics.weight.data)
            print('Singular Values FW:\n',str(S.data.detach().cpu().numpy()))

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

        '''Forward'''
        H_prime = self.dynamics(G.reshape(-1, 1, block_size)).reshape(bs, T, N, D)
        res = H - H_prime
        res, U = res.reshape(bs, T, N, -1, block_size), \
                  U.reshape(bs, T, N, -1, a_block_size)
        res, U = res.permute(0, 3, 1, 2, 4), U.permute(0, 3, 1, 2, 4)

        U = U.reshape(bs, T, N, -1, a_block_size)
        U = U.permute(0, 3, 1, 2, 4)

        B = torch.bmm(self.batch_pinv(U.reshape(bs, T * N, a_block_size), I_factor),
                      res.reshape(bs, T * N, block_size)
                      )

        B_fw = B.reshape(bs, 1, a_block_size, block_size)
        B_bw = None
        return B_fw, B_bw

    def linear_forward_no_input(self, g):
        """
        :param g: B x N x D
        :return:
        """
        ''' B x N x R D '''
        bs, dim = g.shape
        new_g = self.dynamics(g).reshape(bs, dim)
        return new_g

    def linear_backward_no_input(self, g):
        """
        :param g: B x N x D
        :return:
        """
        ''' B x N x R D '''
        bs, dim = g.shape
        new_g = self.backdynamics(g).reshape(bs, dim)
        return new_g

    def linear_forward_wrong(self, g, input):
        """
        :param g: B x N x D
        :return:
        """
        ''' B x N x R D '''
        bs, dim = g.shape
        new_g = (self.dynamics(g) + input).reshape(bs, dim)
        return new_g

    def linear_forward(self, g, u):
        """
        :param g: B x N x D
        :return:
        """
        ''' B x N x R D '''
        bs, dim = g.shape
        new_g = (self.dynamics(g) + self.u_dynamics(g, u, propagate=True)).reshape(bs, dim)
        return new_g

    # def linear_backward(self, g):
    #     """
    #     :param g: B x N x D
    #     :return:
    #     """
    #     ''' B x N x R D '''
    #     bs, dim = g.shape
    #     input, u, u_logit = self.u_dynamics(g)
    #     new_g = (self.backdynamics(g) + input).reshape(bs, dim)
    #     return new_g, u, u_logit

    def rollout_wrong(self, g, u, T, B, backward=False, clip=30, logit_out=False):
        """
        :param g: B x N x D
        :param rel_attrs: B x N x N x R
        :param T:
        :return:
        """
        g = g.squeeze(1)
        g_list = [g[:,None,:]]
        u_list = [] # Note: Realize that first element of u_list corresponds to g(t) and first element of g_list to g(t+1)
        u_logit_list = []
        out_states = [self.inv_mapping(g)[:,None,:]]
        for t in range(T):
            if not backward:
                input, u, u_logit = self.u_dynamics(out_states[t].squeeze(1))
                if len(u_list) == 0:
                    u_list.append(u[:, None, :])
                    u_logit_list.append(u_logit[:, None, :])
                g = self.linear_forward(g, input) # For single object
            # else:
            #     g, u, u_logit = self.linear_backward(g, u)
            out_states.append(self.inv_mapping(g)[:,None,:])
            g_list.append(g[:, None, :])
            u_list.append(u[:, None, :])
            u_logit_list.append(u_logit[:, None, :])
        u_logit = torch.cat(u_logit_list, 1) if logit_out else None
        return torch.cat(out_states, 1),torch.cat(g_list, 1), torch.cat(u_list, 1),  u_logit #torch.clamp(torch.cat(g_list, 1), min=-clip, max=clip)

    def rollout(self, g, u, T, B, backward=False, clip=30, logit_out=False):
        """
        :param g: B x N x D
        :param rel_attrs: B x N x N x R
        :param T:
        :return:
        """
        g = g.squeeze(1)
        g_list = [g[:,None,:]]
        u_list = [] # Note: Realize that first element of u_list corresponds to g(t) and first element of g_list to g(t+1)
        u_logit_list = []
        out_states = [self.inv_mapping(g)[:,None,:]]
        for t in range(T):
            if not backward:
                u, u_logit = self.u_dynamics(out_states[t].squeeze(1), propagate=False)
                if len(u_list) == 0:
                    u_list.append(u[:, None, :])
                    u_logit_list.append(u_logit[:, None, :])
                g = self.linear_forward(g, u) # For single object
            # else:
            #     g, u, u_logit = self.linear_backward(g, u)
            out_states.append(self.inv_mapping(g)[:,None,:])
            g_list.append(g[:, None, :])
            u_list.append(u[:, None, :])
            u_logit_list.append(u_logit[:, None, :])
        u_logit = torch.cat(u_logit_list, 1) if logit_out else None
        return torch.cat(out_states, 1),torch.cat(g_list, 1), torch.cat(u_list, 1),  u_logit #torch.clamp(torch.cat(g_list, 1), min=-clip, max=clip)

    def rollout_no_input(self, g, u, T, B, backward=False, clip=30):
        """
        :param g: B x N x D
        :param rel_attrs: B x N x N x R
        :param T:
        :return:
        """
        g = g.squeeze(1)
        g_list = [g[:,None,:]]
        out_states = [self.inv_mapping(g)[:,None,:]]
        for t in range(T):
            if not backward:
                g = self.linear_forward_no_input(g) # For single object
            else:
                g = self.linear_backward_no_input(g)
            out_states.append(self.inv_mapping(g)[:,None,:])
            g_list.append(g[:, None, :])
        return torch.cat(out_states, 1), torch.cat(g_list, 1), None, None #torch.clamp(torch.cat(g_list, 1), min=-clip, max=clip)


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

