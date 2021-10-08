from abc import ABC

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from model.networks_space.spectral_norm import SpectralNorm
from utils import tracker_util as tut
from utils import util as ut
import math

def gaussian_init_(n_units, std=1):
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std/n_units]))
    Omega = sampler.sample((n_units, n_units))[..., 0]
    return Omega

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
    N = math.sqrt(NN)
    # Assuming poses are in the first and second position.
    mask_I = (torch.eye(N)[None, :, :, None].to(rel_states.device)).reshape(1, NN, 1)
    rel_states = rel_states * mask_I
    current_x = torch.abs(rel_states[..., 0:1])
    current_y = torch.abs(rel_states[..., n_timesteps:n_timesteps+1]) #TODO: Doesn't work with deriv in state
    sel = torch.where((current_x > 2*collision_margin) * (current_y > 2*collision_margin),
                    torch.ones_like(current_x), torch.zeros_like(current_x))
    return sel

'''My definition'''

class Embedding(nn.Module):
    def __init__(self, s, c, hidden_size, g, SN=False):
        super(Embedding, self).__init__()
        self.g = g
        self.c = c
        self.model = nn.Sequential(
            nn.Linear(s, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, (g * c) + 1),
        )

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
        # x = torch.clamp(x, min=-1.1, max=1.1)
        o = self.model(x)
        g_sel_logit = o[..., :1]
        g_out = o[..., 1:].reshape(-1, self.c, self.g)
        return g_out, g_sel_logit

class U_Embedding(nn.Module):
    def __init__(self, s, c, hidden_size, output_size, collision_margin=None, SN=False):
        super(U_Embedding, self).__init__()
        self.collision_margin = collision_margin
        self.output_size = output_size
        self.c = c
        self.u_dim = output_size
        self.sigmoid = nn.Sigmoid()
        self.gumbel_sigmoid = tut.STGumbelSigmoid()
        # Input is s_o1 - s_o2
        # if collision_margin is None:
        self.model_u = nn.Sequential(
            nn.Linear(s, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

        self.model = nn.Sequential(
            nn.Linear(s, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size * c),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        self.round = tut.Round()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x, temp=1, n_timesteps = None, collision_margin=None):
        """
        Args:
            x: [n_relations, state_size] (relative states)
        Returns:
            [n_relations, output_size]
            [n_relations, selector_size]
        """
        BT, N, sd = x.shape
        u_logits = self.model_u(x)
        u_enc = self.model(x).tanh()

        if collision_margin is None:
            # u = torch.exp(u_logits)
            u = (u_logits / temp).sigmoid()
            # u = self.gumbel_sigmoid(u)
        else:
            u_logits = None
            print('Not now')
            exit()
            u = get_u_states_mask(x, collision_margin, n_timesteps)

        g_u = (u_enc).reshape(BT, N, -1) #* u
        g_u = g_u.reshape(-1, self.c, self.u_dim)

        return g_u, u, u_logits

class Rel_Embedding(nn.Module):
    def __init__(self, s, c, hidden_size, output_size, collision_margin=None, SN=False):
        super(Rel_Embedding, self).__init__()
        self.collision_margin = collision_margin
        self.output_size = output_size
        self.c = c
        self.g = output_size
        self.sigmoid = nn.Sigmoid()
        self.gumbel_sigmoid = tut.STGumbelSigmoid()
        # Input is s_o1 - s_o2
        # if collision_margin is None:

        self.model_rel = nn.Sequential(
            nn.Linear(3*s, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

        self.model = nn.Sequential(
            nn.Linear(3*s, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size * c),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        # if SN:
        #     self.model = SpectralNorm(self.model)
        #     self.model_rel = SpectralNorm(self.model_u)

        self.round = tut.Round()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x, rel_x, temp=1, n_timesteps = None, collision_margin=None):
        """
        Args:
            x: [n_relations, state_size] (relative states)
        Returns:
            [n_relations, output_size]
            [n_relations, selector_size]
        """

        # TODO: There should be an interaction matrix if masses are all the same.
        BT, N, sd = x.shape
        sel_logits = self.model_rel(rel_x)
        rel_enc = self.model(rel_x)
        if collision_margin is None:
            # sel = torch.exp(sel_logits)
            sel = sel_logits.sigmoid()
            sel = self.gumbel_sigmoid(sel)
        else:
            sel_logits = None
            print('Not now')
            sel = get_rel_states_mask(rel_x.chunk(3, -1)[-1], collision_margin, n_timesteps)

        mask_I = (- torch.eye(N)[None, :, :, None].to(x.device) + 1.0).reshape(1, N*N, 1)
        g_rel = torch.sum((rel_enc * mask_I).reshape(BT, N, N, -1), 2)
        g_rel = g_rel.reshape(-1, self.c, self.g)
        return g_rel, sel, sel_logits

class ObservableDecoderNetwork(nn.Module):
    def __init__(self, s, c, hidden_size, output_size, SN=False):
        super(ObservableDecoderNetwork, self).__init__()

        # self.fc = SpectralNorm(nn.Linear(c, 1))
        # self.model = nn.Sequential(
        #     SpectralNorm(nn.Linear(s, hidden_size)),
        #     nn.ReLU(),
        #     SpectralNorm(nn.Linear(hidden_size, hidden_size)),
        #     nn.ReLU(),
        #     SpectralNorm(nn.Linear(dhidden_size, hidden_size)),
        #     nn.ReLU(),
        #     SpectralNorm(nn.Linear(hidden_size, output_size +1))
        # )
        self.model = nn.Sequential(
            nn.Linear(s, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size + 1)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        B, N, T, c_d, g_d = x.shape
        """
        Args:
            x: [n_particles, input_size]
        Returns:
            [n_particles, output_size]
        """
        x = x.reshape(B*N*T, c_d, g_d)
        # x = self.fc(x.transpose(-2, -1)).squeeze(-1)
        x = x.sum(1)

        o = self.model(x).reshape(B, N, T, -1)
        confi = o[..., -1:]#.sigmoid()
        # out = torch.clamp(o[..., :-1], min=-1.1, max=1.1)
        # out = o[..., :-1].tanh()
        out = o[..., :-1]
        return out, confi

class EncoderNetwork(nn.Module):
    def __init__(self, input_dim, nf_particle, nf_effect, g_dim, r_dim, u_dim, n_chan, with_interactions=False, residual=False, collision_margin=None, n_timesteps=None):
        #TODO: added r_dim
        super(EncoderNetwork, self).__init__()

        # eo_dim = nf_particle
        # es_dim = nf_particle
        self.with_interactions = with_interactions
        self.collision_margin = collision_margin
        self.n_timesteps = n_timesteps

        '''Encoder Networks'''
        self.embedding = Embedding(input_dim, n_chan, hidden_size=nf_particle, g=g_dim)
        self.u_embedding = U_Embedding(input_dim, n_chan, hidden_size=nf_effect, output_size=u_dim, collision_margin=collision_margin)

        if with_interactions:
            self.rel_embedding = Rel_Embedding(input_dim, n_chan, hidden_size=nf_effect, output_size=r_dim, collision_margin=collision_margin)

        # TODO: There's the Gumbel Softmax, if this wasn't working. Check the Tracking code if they do KL divergence in that case
        self.softmax = nn.Softmax(dim=-1)
        self.round = tut.Round()
        self.st_gumbel_softmax = tut.STGumbelSoftmax(-1)

    def forward(self, states, temp, with_u=True, inputs=None, n_timesteps=None, collision_margin=None):

        B, N, T, sd = states.shape
        BT = B*T
        states = states.permute(0, 2, 1, 3) # [B, T, N, sd]
        states = states.reshape(BT, N, sd) # [B * T, N, sd]

        g, g_sel_logits = self.embedding(states)
        if with_u:
            g_u, u, u_logits = self.u_embedding(states)
        else:
            g_u, u, u_logits = None, None, None

        if inputs is not None:
            u = inputs

        if self.with_interactions:
            # TODO: !!!! exchange predictions for objects. o1-o2 --> o2-o1.
            #  This is the same as transposing randomly the matrix of relative obj. And making sel symmetric.
            # TODO: hardcodeinteractions in the rel_encoder. check what are pose distances.
            #  do the same with u? We assume to know limits of the frame.

            states_receiver = states[:, :, None, :].repeat(1, 1, N, 1).reshape(BT, N*N, -1)
            states_sender = states[:, None, :, :].repeat(1, N, 1, 1).reshape(BT, N*N, -1)
            rel_states = (states_sender - states_receiver).reshape(BT, N*N, -1)

            g_rel, sel, sel_logits = self.rel_embedding(states, torch.cat([states_receiver, states_sender, rel_states], dim=-1), temp=temp, n_timesteps=self.n_timesteps, collision_margin=collision_margin)

            if sel is not None:
                sel = sel.reshape(B, T, N, N).permute(0, 2, 1, 3).reshape(B, N, T, N) # We show the selector for every object in dim=-2
            if sel_logits is not None:
                sel_logits = sel_logits.reshape(B, T, N, N).permute(0, 2, 1, 3).reshape(B, N, T, N)

            # TODO: Implement with hardcoded selectors
            mask = g_sel_logits
            # mask = self.round(self.softmax(mask/temp))
            # mask = self.softmax(mask)
            # tut.norm_grad(mask, 10)
            # mask = self.st_gumbel_softmax(mask)
            obs = g

        else:
            # mask = torch.cat([g_sel_logits, u_logits], dim=-1)
            mask = g_sel_logits
            # mask = self.round(self.softmax(mask/temp))
            # mask = self.softmax(mask/temp)

            obs = g
            sel, sel_logits = None, None


        # TODO: Bonissima.
        #  Idea 1: Deixar utilitzar comp_obs nomes x time_steps en training, despres es tot obs. Aixi hi ha backprop
        #  Idea 2: El mateix amb Ag en tots menys 1
        #  Idea 3: El primer step es sempre amb A*obs.

        '''Disclamers'''
        # TODO: Here would originally come another net to unite them all. Particle predictor.
        # Note: Rel Attrs is the relation attributes such as spring constant. In free space we ignore them.
        # TODO: We should add a network that maps noise (or ones) to something. Like gravitational constant.
        #  If we don't, we have to let know that we assume no external interaction but the objects collision to the environment.


        obs = obs.reshape(B, T, N, *obs.shape[-2:]).permute(0, 2, 1, 3, 4).reshape(B, N, T, *obs.shape[-2:])

        if g_u is not None:
            g_u = g_u.reshape(B, T, N, *g_u.shape[-2:]).permute(0, 2, 1, 3, 4).reshape(B, N, T, *g_u.shape[-2:])

        # Note: We punish directly g_u
        if u is not None and g_u is not None:
            u   =  g_u  .reshape(B, T, N, -1).permute(0, 2, 1, 3).reshape(B, N, T, -1)
        if u_logits is not None:
            u_logits = u_logits.reshape(B, T, N, -1).permute(0, 2, 1, 3).reshape(B, N, T, -1)

        selector = mask.reshape(B, T, N, -1).permute(0, 2, 1, 3).reshape(B, N, T, -1)
        return obs, g_u, (u, u_logits), (sel, sel_logits), selector

class dynamics(nn.Module):
    def __init__(self, g, r, u, with_interactions, init_scale):
        super(dynamics, self).__init__()

        # TODO: Doubts about the size. Different matrix for each interaction? We start by aggregating relation_states and using same matrix?
        # TODO: Maybe get all states and sum them with a matrix

        #Option 1

        self.u_dynamics = nn.Linear(u, g, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
        # eig_vec = torch.eig(self.dynamics.weight.data, eigenvectors=True)[1]
        # self.dynamics.weight.data = eig_vec
        U, S, V = torch.svd(self.u_dynamics.weight.data)
        self.u_dynamics.weight.data = torch.mm(U, V.t()) * 0.8

        #Option 2
        self.dynamics = nn.Linear(g, g, bias=False)
        self.dynamics.weight.data = torch.zeros_like(self.dynamics.weight.data) + 0.001

        # self.dynamics.weight.data = gaussian_init_(g, std=0.1)
        # U, S, V = torch.svd(self.dynamics.weight.data)
        # self.dynamics.weight.data = torch.mm(U, V.t()) * 0.7

        #Option 3

        # k = 5
        # S = torch.ones_like(S)
        # S[..., k:] = 0
        # self.dynamics.weight.data = torch.matmul(torch.matmul(U, torch.diag_embed(S)), V.t()) * init_scale
        # TODO: Recursively keep only k singular values.
        # TODO: if OLS G = g[:-1], H= g[1:], svd(g). --> Might not be right because you have to invert G and not H

    def forward(self, g, u):
        g = self.dynamics(g)
        if u is not None:
            g = g + self.u_dynamics(u)
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

        n_chan = 1

        input_dim = state_dim * (n_timesteps + first_deriv_dim + sec_deriv_dim) #+ g_dim # g_dim added for recursive sampling

        '''Encoder Network'''
        self.mapping = EncoderNetwork(input_dim, nf_particle, nf_effect, g_dim, r_dim, u_dim, n_chan, with_interactions=with_interactions, residual=False, collision_margin=collision_margin, n_timesteps=n_timesteps)

        '''Decoder Network'''
        self.inv_mapping = ObservableDecoderNetwork(g_dim, n_chan, hidden_size=nf_particle, output_size=input_dim, SN=False)

        '''Koopman operator'''
        self.dynamics = dynamics(g_dim, r_dim, u_dim, with_interactions, init_scale)

        self.simulate = self.rollout_B

    def to_s(self, gcodes, psteps):
        """ state decoder """
        states, confi = self.inv_mapping(gcodes)
        return states, confi

    def to_g(self, states, temp=1, inputs=None, with_u=True):
        """ state encoder """
        return self.mapping(states, temp=temp, collision_margin=self.collision_margin, inputs=inputs, with_u=with_u)

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

    def get_AB(self):
        A = self.dynamics.dynamics.weight.data
        B = self.dynamics.u_dynamics.weight.data
        return A, B

    def add_noise_to_A(self, std=0.1):
        A = self.dynamics.dynamics.weight.data
        self.dynamics.dynamics.weight.data = A + torch.randn_like(A)*std / (A.shape[0])

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
            A = self.dynamics.dynamics.weight.data
            # A_I = torch.eye(A.shape[0]).to(A.device)
            A_I = torch.zeros_like(A)
            U, S, V = torch.svd(A + A_I)
            E = torch.eig(A + A_I)[0]
            mod_E = torch.norm(E, dim=-1)
            print('Singular Values FW:\n',str(S.data.detach().cpu().numpy()),'\nEigenvalues FW:\n',str(mod_E.data.detach().cpu().numpy()))

    def linear_forward(self, g, u=None):
        """
        :param g: B x N x D
        :return:
        """
        ''' B x N x R D '''
        B, N, T, c, dim = g.shape
        g_in = g.reshape(B*N*T, *g.shape[-2:])
        if u is not None:
            u = u.reshape(B*N*T, *u.shape[-2:])
        new_g = (self.dynamics(g_in, u)).reshape(B, N, T, *g.shape[-2:])
        # new_g.register_hook(lambda x: torch.clamp(x, min=-0, max=0))
        # new_g.register_hook(lambda x: print(x.norm()))
        return new_g

    def rollout_in_out(self, g, T, init_s, inputs=None, temp=1, logit_out=False):
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
        selector_list = []
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
            g_stack, u_tp, sel_tp, selector = self.mapping(in_full_state, temp=temp, collision_margin=self.collision_margin, input=in_t)
            u, u_logits = u_tp
            sel, sel_logits = sel_tp
            selector_list.append(selector)
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
            # g_out = self.linear_forward(g_stack.transpose(-2, -1).flatten(start_dim=-2))
            g_out = self.rollout_1step(g_stack)

            out_states.append(self.inv_mapping(g_out))
            g_list.append(g_out)
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
        u_list = torch.cat(u_list, 2) if u_list[0] is not None else None
        u_logit = torch.cat(u_logit_list, 2) if logit_out and u_logit_list[0] is not None else None

        selectors = torch.cat(selector_list, 2)

        return torch.cat(out_states, 2), torch.cat(g_list, 2), (u_list,  u_logit), (sel_list, sel_logit_list), selectors

    def rollout(self, g, T, inputs=None, temp=1, logit_out=False, with_u = True):
        """
        :param g: B x N x D
        :param rel_attrs: B x N x N x R
        :param T:
        :return:
        """
        # TODO: Test function.
        B, N, _, gs, ns = g.shape
        g_list = []

        u_list = [] # Note: Realize that first element of u_list corresponds to g(t) and first element of g_list to g(t+1)
        u_logit_list = []
        sel_list = []
        sel_logit_list = []
        selector_list = []
        if inputs is None:
            in_t = None

        out_s, out_c = self.inv_mapping(g)
        out_states = [out_s]
        out_confis = [out_c]

        for t in range(T):
            if inputs is not None:
                in_t = inputs[..., t, :]
            g_in, u_in, u_tp, sel_tp, selector = self.mapping(out_states[t], temp=temp, collision_margin=self.collision_margin, inputs=in_t, with_u = with_u)
            u, u_logits = u_tp
            sel, sel_logits = sel_tp
            selector_list.append(selector)

            if len(u_list) == 0:
                u_list.append(u)
                u_logit_list.append(u_logits)
                # input_list.append(inputs[:, None, :])
            if self.with_interactions and len(sel_list)==0:
                sel_list.append(sel)
                sel_logit_list.append(sel_logits)

            if t == 0:
                g_list.append(g_in)
            # Propagate with A
            g_out = self.rollout_1step(g_in, u_in)
            g_list.append(g_out)
            out_s, out_c = self.inv_mapping(g_out)
            out_states.append(out_s)
            out_confis.append(out_c)

        gs = torch.cat(g_list, 2)
        out_state = torch.cat(out_states, 2)
        out_confi = torch.cat(out_confis, 2)
        return out_state, out_confi, gs, (None,  None), (None, None), None

    def rollout_B(self, g, T, inputs=None, temp=1, logit_out=False, with_u = True):
        """
        :param g: B x N x D
        :param rel_attrs: B x N x N x R
        :param T:
        :return:
        """
        # TODO: Test function.
        B, N, _, gs, ns = g.shape
        g_list = []

        u_list = [] # Note: Realize that first element of u_list corresponds to g(t) and first element of g_list to g(t+1)
        u_logit_list = []
        sel_list = []
        sel_logit_list = []
        selector_list = []
        if inputs is None:
            in_t = None

        out_s, out_c = self.inv_mapping(g)
        out_states = [out_s]
        out_confis = [out_c]

        g_in = g
        for t in range(T):
            if inputs is not None:
                in_t = inputs[..., t, :]
            _, u_in, u_tp, sel_tp, selector = self.mapping(out_states[t], temp=temp, collision_margin=self.collision_margin, inputs=in_t, with_u = with_u)
            u, u_logits = u_tp
            sel, sel_logits = sel_tp
            selector_list.append(selector)

            if len(u_list) == 0:
                u_list.append(u)
                u_logit_list.append(u_logits)
                # input_list.append(inputs[:, None, :])
            if self.with_interactions and len(sel_list)==0:
                sel_list.append(sel)
                sel_logit_list.append(sel_logits)

            if t == 0:
                g_list.append(g_in)
            # Propagate with A
            g_in = self.rollout_1step(g_in, u_in)
            g_list.append(g_in)
            out_s, out_c = self.inv_mapping(g_in)
            out_states.append(out_s)
            out_confis.append(out_c)

        gs = torch.cat(g_list, 2)
        out_state = torch.cat(out_states, 2)
        out_confi = torch.cat(out_confis, 2)
        return out_state, out_confi, gs, (None,  None), (None, None), None

    def rollout_1step(self, g_stack, u = None):
        """
        :param g: B x N x D
        :param rel_attrs: B x N x N x R
        :param T:
        :return:
        """
        if len(g_stack.shape) == 5:
            B, N, T, c, gs = g_stack.shape
            g_in = g_stack
        else:
            raise NotImplementedError

        g_out = self.linear_forward(g_in, u).reshape(B, N, T, c, gs) #+ g_in

        return g_out


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
            new_x.append(torch.stack([x[:, t + (self.n_timesteps-idx-1)]
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
