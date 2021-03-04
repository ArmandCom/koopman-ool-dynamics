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

# def gaussian_init_2dim(units, std=1):
#     assert len(units) == 2
#     sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std/units[0]]))
#     Omega = sampler.sample((units[0], units[1]))[..., 0]
#     return Omega

# TODO: Decide if OLS (flexible with num obj). Same A for all?

def get_u_states_mask(states, collision_margin, n_timesteps):
    B, N, sd = states.shape
    # Assuming poses are in the first and second position.
    current_x = torch.abs(states[..., 0:1])
    current_y = torch.abs(states[..., n_timesteps:n_timesteps+1])
    u = torch.where((current_x > 1 - collision_margin) + (current_y > 1 - collision_margin),
                    torch.ones_like(current_x), torch.zeros_like(current_x))
    return u

def get_rel_states_mask(rel_states, collision_margin, n_timesteps):
    B, NN, rsd = rel_states.shape
    # Assuming poses are in the first and second position.
    current_x = torch.abs(rel_states[..., 0:1])
    current_y = torch.abs(rel_states[..., n_timesteps:n_timesteps+1]) #TODO: Doesn't work with deriv in state
    sel = torch.where((current_x > 2*collision_margin) * (current_y > 2*collision_margin),
                    torch.ones_like(current_x), torch.zeros_like(current_x))
    return sel

'''My definition'''
class Rel_Embedding(nn.Module):
    def __init__(self, s, rel_size, hidden_size, output_size, collision_margin=None, SN=False):
        super(Rel_Embedding, self).__init__()
        self.collision_margin = collision_margin
        self.output_size = output_size
        # Input is s_o1 - s_o2
        # if collision_margin is None:
        self.model_rel = nn.Sequential(
            nn.Linear(3*s, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, rel_size + 1),
        )

        self.model = nn.Sequential(
            nn.Linear(s + rel_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

        if SN:
            self.model = SpectralNorm(self.model)
            self.model_rel = SpectralNorm(self.model_u)

        self.round = tut.Round()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x, rel_x, temp=1, n_timesteps = None, collision_margin=None):
        """
        Args:
            x: [n_relations, state_size] (relative states)
        Returns:
            [n_relations, output_size]
            [n_relations, selector_size]
        """
        BT, N, sd = x.shape
        o = self.model_rel(rel_x)
        rel_enc = self.relu(o[..., :-1])
        if collision_margin is None:
            sel_logits = o[..., -1:]

            # Note: Hard determin.
            sel = self.round(self.sigmoid(sel_logits / temp))

            # Note: Soft determin
            # sel = self.sigmoid(sel_logits / temp)

            # Note: Soft Stochastic
            # if noise and self.count<8000:
            #     sel_logit = sel_logits + torch.randn(sel_logits.shape).to(u_logits.device)*0.1
            #     self.count+=1

            # sel_logits = sel_logits.tanh() * 8.8
            # sel_post = ut.NumericalRelaxedBernoulli(logits=sel_logits, temperature=temp)
            # # Unbounded
            # sel = sel_post.rsample()
            # sel = torch.sigmoid(sel / temp)

        else:
            sel_logits = None
            sel = get_rel_states_mask(rel_x.chunk(3, -1)[-1], collision_margin, n_timesteps)

        mask_I = (- torch.eye(N)[None, :, :, None].to(x.device) + 1.0).reshape(1, N*N, 1)
        sel = sel * mask_I
        rel_enc = (rel_enc * sel).reshape(BT, N, N, -1).sum(2)
        g = self.model(torch.cat([x, rel_enc], dim=-1))

        return g, sel, sel_logits

class Embedding(nn.Module):
    def __init__(self, s, hidden_size, g, SN=False):
        super(Embedding, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(s, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, g),
        )
        if SN:
            self.model = SpectralNorm(self.model)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        """
        Args:
            x: [n_particles, input_size]
        Returns:
            [n_particles, output_size]
        """
        return self.model(x)

class U_Embedding(nn.Module):
    def __init__(self, input_size, u_size, hidden_size, output_size, SN=False):
        super(U_Embedding, self).__init__()

        self.input_size = input_size
        self.u_size = u_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.model_u = nn.Sequential(
            nn.Linear(input_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, u_size + 1),
        )
        self.model = nn.Sequential(
            nn.Linear(input_size + u_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

        if SN:
            self.model = SpectralNorm(self.model)
            self.model_u = SpectralNorm(self.model_u)
        #TODO: adopt Sandesh's SpectralNorm.


        self.round = tut.Round()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x, temp=1):
        """
        Args:
            x: [n_particles, input_size]
        Returns:
            [n_particles, output_size]
        """
        o = self.model_u(x)
        inputs, u_logits = o[..., :-1], o[..., -1:]
        inputs = self.relu(inputs)
        u = self.round(self.sigmoid(u_logits/temp))
        x = self.model(torch.cat([x, inputs], dim=-1)) * u
        return x, u, u_logits

class ObservableDecoderNetwork(nn.Module):
    def __init__(self, s, hidden_size, output_size, SN=False):
        super(ObservableDecoderNetwork, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(s, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        if SN:
            self.model = SpectralNorm(self.model)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        B, N, T, _ = x.shape
        """
        Args:
            x: [n_particles, input_size]
        Returns:
            [n_particles, output_size]
        """
        o = self.model(x.reshape(B*N*T, -1)).reshape(B, N, T, -1)
        o = torch.clamp(o, min=-1.1, max=1.1) # TODO: make sure this works well.

        return o

class EncoderNetwork(nn.Module):
    def __init__(self, input_dim, nf_particle, nf_effect, g_dim, r_dim, u_dim, with_interactions=False, residual=False, collision_margin=None, n_timesteps=None):
        #TODO: added r_dim
        super(EncoderNetwork, self).__init__()

        # eo_dim = nf_particle
        # es_dim = nf_particle
        self.with_interactions = with_interactions
        self.collision_margin = collision_margin
        self.n_timesteps = n_timesteps

        '''Encoder Networks'''
        self.u_embedding = U_Embedding(input_dim, u_size=u_dim, hidden_size=nf_effect, output_size=g_dim)
        self.embedding = Embedding(input_dim, hidden_size=nf_particle, g=g_dim)

        if with_interactions:
            self.rel_embedding = Rel_Embedding(input_dim, rel_size=r_dim, hidden_size=nf_effect, output_size=g_dim, collision_margin=collision_margin)

    def forward(self, states, temp, input=None, n_timesteps=None, collision_margin=None):

        B, N, T, sd = states.shape
        BT = B*T
        states = states.permute(0, 2, 1, 3) # [B, T, N, sd]
        states = states.reshape(BT, N, sd) # [B * T, N, sd]

        g_u, u, u_logits = self.u_embedding(states, temp=temp)
        g = self.embedding(states)

        if self.with_interactions:
            # TODO: !!!! exchange predictions for objects. o1-o2 --> o2-o1.
            #  This is the same as transposing randomly the matrix of relative obj. And making sel symmetric.
            # TODO: hardcodeinteractions in the rel_encoder. check what are pose distances.
            #  do the same with u? We assume to know limits of the frame.

            states_receiver = states[:, :, None, :].repeat(1, 1, N, 1).reshape(BT, N*N, -1)
            states_sender = states[:, None, :, :].repeat(1, N, 1, 1).reshape(BT, N*N, -1)
            rel_states = (states_sender - states_receiver).reshape(BT, N*N, -1)

            g_rel, sel, sel_logits = self.rel_embedding(states, torch.cat([states_receiver, states_sender, rel_states], dim=-1), temp=temp, n_timesteps=self.n_timesteps, collision_margin=collision_margin)

            sel_max = torch.clamp(sel.reshape(BT, N, N, -1).sum(2), min=0, max=1)
            g_rel = g_rel * sel_max

            sel = sel.reshape(B, T, N, N).permute(0, 2, 1, 3).reshape(B, N, T, N) # We show the selector for every object in dim=-2
            if sel_logits is not None:
                sel_logits = sel_logits.reshape(B, T, N, N).permute(0, 2, 1, 3).reshape(B, N, T, N)
        else:
            sel_max = 0
            g_rel = 0
            sel, sel_logits = None, None

        g = g * (1-u) * (1-sel_max)
        obs = g_rel + g_u + g

        # TODO: Bonissima.
        #  Idea 1: Deixar utilitzar comp_obs nomes x time_steps en training, despres es tot obs. Aixi hi ha backprop
        #  Idea 2: El mateix amb Ag en tots menys 1
        #  Idea 3: El primer step es sempre amb A*obs.

        '''Disclamers'''
        # TODO: Here would originally come another net to unite them all. Particle predictor.
        # Note: Rel Attrs is the relation attributes such as spring constant. In free space we ignore them.
        # TODO: We should add a network that maps noise (or ones) to something. Like gravitational constant.
        #  If we don't, we have to let know that we assume no external interaction but the objects collision to the environment.

        obs = obs.reshape(B, T, N, -1).permute(0, 2, 1, 3).reshape(B, N, T, -1)
        u   = u  .reshape(B, T, N, -1).permute(0, 2, 1, 3).reshape(B, N, T, -1)
        if u_logits is not None:
            u_logits = u_logits.reshape(B, T, N, -1).permute(0, 2, 1, 3).reshape(B, N, T, -1)

        return g, g_u, g_rel, (u, u_logits), (sel, sel_logits)

class dynamics(nn.Module):
    def __init__(self, g, r, u, init_scale):
        super(dynamics, self).__init__()

        # TODO: Doubts about the size. Different matrix for each interaction? We start by aggregating relation_states and using same matrix?
        # TODO: Maybe get all states and sum them with a matrix
        self.dynamics = nn.Linear(g, g, bias=False)

        #Option 1

        self.dynamics.weight.data = torch.zeros_like(self.dynamics.weight.data) + 0.001

        #Option 2

        # self.dynamics.weight.data = gaussian_init_(g, std=1)
        # U, S, V = torch.svd(self.dynamics.weight.data)
        # self.dynamics.weight.data = torch.mm(U, V.t()) * 0.1

        #Option 3

        # k = 5
        # S = torch.ones_like(S)
        # S[..., k:] = 0
        # self.dynamics.weight.data = torch.matmul(torch.matmul(U, torch.diag_embed(S)), V.t()) * init_scale
        # TODO: Recursively keep only k singular values.
        # TODO: if OLS G = g[:-1], H= g[1:], svd(g). --> Might not be right because you have to invert G and not H

    def forward(self, g):
        # x = torch.cat([g, rel, u], dim=-1)
        g = self.dynamics(g)
        return g

class KoopmanOperators(nn.Module, ABC):
    def __init__(self, state_dim, nf_particle, nf_effect, g_dim, r_dim, u_dim, n_timesteps, deriv_in_state=False, with_interactions = False, init_scale=1, collision_margin=None):

        super(KoopmanOperators, self).__init__()

        self.n_timesteps = n_timesteps
        self.collision_margin = collision_margin
        self.deriv_in_state = deriv_in_state
        self.u_dim = u_dim
        self.with_interactions = with_interactions

        if deriv_in_state and n_timesteps > 2:
            first_deriv_dim = n_timesteps - 1
            sec_deriv_dim = n_timesteps - 2
        else:
            first_deriv_dim = 0
            sec_deriv_dim = 0

        input_dim = state_dim * (n_timesteps + first_deriv_dim + sec_deriv_dim) #+ g_dim # g_dim added for recursive sampling

        '''Encoder Network'''
        self.mapping = EncoderNetwork(input_dim, nf_particle, nf_effect, g_dim, r_dim, u_dim, with_interactions=with_interactions, residual=False, collision_margin=collision_margin, n_timesteps=n_timesteps)

        '''Decoder Network'''
        self.inv_mapping = ObservableDecoderNetwork(g_dim, hidden_size=nf_particle, output_size=state_dim)

        '''Koopman operator'''
        self.dynamics = dynamics(g_dim, r_dim, u_dim, init_scale)

        self.simulate = self.rollout

    def to_s(self, gcodes, psteps):
        """ state decoder """
        states = self.inv_mapping(gcodes)
        return states

    def to_g(self, states, temp=1, input=None):
        """ state encoder """
        return self.mapping(states, temp=temp, collision_margin=self.collision_margin, input=input)

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

    def get_A(self):
        A = self.dynamics.dynamics.weight.data
        return A

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

    def linear_forward(self, g):
        """
        :param g: B x N x D
        :return:
        """
        ''' B x N x R D '''
        B, N, T, dim = g.shape
        new_g = (self.dynamics(g.reshape(B*N*T, -1))).reshape(B, N, T, dim)
        return new_g

    def rollout(self, g, T, init_s, inputs=None, temp=1, logit_out=False):
        """
        :param g: B x N x D
        :param rel_attrs: B x N x N x R
        :param T:
        :return:
        """
        # TODO: Test function.
        B, N, _, gs = g.shape
        g_list = [g]
        u_list = [] # Note: Realize that first element of u_list corresponds to g(t) and first element of g_list to g(t+1)
        u_logit_list = []
        sel_list = []
        sel_logit_list = []
        if inputs is None:
            in_t = None

        out_states = [self.inv_mapping(g)]
        in_states = [torch.cat([init_s, out_states[-1]], dim=-2)]
        # ut.print_var_shape(out_states[0], 's in rollout')

        for t in range(T):
            in_full_state, _ = self.get_full_state_hankel(in_states[-1].reshape(B*N, self.n_timesteps, -1),
                                                          T=self.n_timesteps)
            in_full_state = in_full_state.reshape(B, N, 1, -1)
            # TODO: check correspondence between g and in_states
            if inputs is not None:
                in_t = inputs[..., t, :]
            g, u_tp, sel_tp = self.mapping(in_full_state, temp=temp, collision_margin=self.collision_margin, input=in_t)
            u, u_logits = u_tp
            sel, sel_logits = sel_tp
            if u is None:
                logit_out = True

            if len(u_list) == 0:
                u_list.append(u)
                u_logit_list.append(u_logits)
                # input_list.append(inputs[:, None, :])
            if self.with_interactions and len(sel_list)==0:
                sel_list.append(sel)
                sel_logit_list.append(sel_logits)

            # Propagate with A
            g = self.linear_forward(g)

            out_states.append(self.inv_mapping(g))
            g_list.append(g)
            in_states.append(torch.cat([in_states[-1][..., 1:, :], out_states[-1]], dim=-2))
            u_list.append(u)
            u_logit_list.append(u_logits)
            # input_list.append(inputs[:, None, :])

        if self.with_interactions:
                sel_list.append(sel)
                sel_logit_list.append(sel_logits)
        if self.with_interactions:
            sel_list = torch.cat(sel_list, 2)
            sel_logit_list = torch.cat(sel_logit_list, 2) if logit_out else None
        else:
            sel_list, sel_logit_list = None, None
        u_logit = torch.cat(u_logit_list, 2) if logit_out else None

        return torch.cat(out_states, 2), torch.cat(g_list, 2), (torch.cat(u_list, 2),  u_logit), (sel_list, sel_logit_list)

    def get_full_state_hankel(self, x, T):
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

        if self.deriv_in_state and self.n_timesteps > 2:
            d_x = new_x[..., 1:] - new_x[..., :-1]
            dd_x = d_x[..., 1:] - d_x[..., :-1]
            new_x = torch.cat([new_x, d_x, dd_x], dim=-1)

        # print(new_x[0, 0, :4, :3], new_x.flatten(start_dim=-2)[0, 0, :12])
        # exit() out: [x1, x2, x3, y1, y2, y3]
        return new_x.flatten(start_dim=-2), new_T

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

# class encoderNet(nn.Module):
#     def __init__(self, N, b, ALPHA = 1):
#         super(encoderNet, self).__init__()
#         self.N = N
#         self.tanh = nn.Tanh()
#         self.b = b
#
#         self.fc1 = nn.Linear(self.N, 16*ALPHA)
#         self.fc2 = nn.Linear(16*ALPHA, 16*ALPHA)
#         self.fc3 = nn.Linear(16*ALPHA, b)
#
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_normal_(m.weight)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0.0)
#
#     def forward(self, x):
#         x = x.view(-1, 1, self.N)
#         x = self.tanh(self.fc1(x))
#         x = self.tanh(self.fc2(x))
#         x = self.fc3(x).view(-1, self.b)
#
#         return x
#
# class decoderNet(nn.Module):
#     def __init__(self, N, b, ALPHA = 1, SN=False):
#         super(decoderNet, self).__init__()
#
#         self.N = N
#         self.b = b
#
#         self.tanh = nn.Tanh()
#         self.relu = nn.ReLU()
#
#         if SN:
#             self.fc1 = SpectralNorm(nn.Linear(b, 16*ALPHA))
#             self.fc2 = SpectralNorm(nn.Linear(16*ALPHA, 16*ALPHA))
#             self.fc3 = SpectralNorm(nn.Linear(16*ALPHA, N))
#         else:
#             self.fc1 = nn.Linear(b, 16*ALPHA)
#             self.fc2 = nn.Linear(16*ALPHA, 16*ALPHA)
#             self.fc3 = nn.Linear(16*ALPHA, N)
#
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_normal_(m.weight)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0.0)
#
#     def forward(self, x):
#         x = x.view(-1, 1, self.b)
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         x = self.fc3(x)
#         x = x.view(-1, self.N)
#         return x
