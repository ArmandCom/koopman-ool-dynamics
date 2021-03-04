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
    current_y = torch.abs(rel_states[..., n_timesteps:n_timesteps+1])
    sel = torch.where((current_x > 2*collision_margin) + (current_y > 2*collision_margin),
                    torch.ones_like(current_x), torch.zeros_like(current_x))
    return sel

'''My definition'''
class RelationEncoder(nn.Module):
    def __init__(self, s, hidden_size, output_size, collision_margin=None):
        super(RelationEncoder, self).__init__()
        self.collision_margin = collision_margin
        self.output_size = output_size
        # Input is s_o1 - s_o2
        if collision_margin is None:
            self.model = nn.Sequential(
                nn.Linear(s, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size + 1),
            )
        else:
            self.model = nn.Sequential(
                nn.Linear(s, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size),
            )

        self.round = tut.Round()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x, temp=1, n_timesteps = None, collision_margin=None):
        """
        Args:
            x: [n_relations, state_size] (relative states)
        Returns:
            [n_relations, output_size]
            [n_relations, selector_size]
        """
        o = self.model(x)


        er = self.relu(o[..., :self.output_size])

        if collision_margin is None:
            sel_logits = o[..., -1:]
            # TODO: implement stochastic.

            # Note: Hard determin.
            # sel = self.round(self.sigmoid(sel_logits / temp))

            # Note: Soft determin
            # sel = self.sigmoid(sel_logits / temp)

            # Note: Soft Stochastic
            # if noise and self.count<8000:
            #     sel_logit = sel_logits + torch.randn(sel_logits.shape).to(u_logits.device)*0.1
            #     self.count+=1

            sel_logits = sel_logits.tanh() * 8.8
            sel_post = ut.NumericalRelaxedBernoulli(logits=sel_logits, temperature=temp)
            # Unbounded
            sel = sel_post.rsample()
            sel = torch.sigmoid(sel / temp)

        else:
            sel_logits = None
            sel = get_rel_states_mask(x, collision_margin, n_timesteps)

        # Mask only relevant relations states? At this moment this is done in RelationPropagator
        # er = er * s

        return er, sel, sel_logits

class RelationPropagator(nn.Module):
    def __init__(self, es, rd, output_size, residual=False):
        super(RelationPropagator, self).__init__()

        self.residual = residual
        self.linear = nn.Linear(2 * es + rd, output_size)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

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

class StateEncoder(nn.Module):
    def __init__(self, s, hidden_size, output_size):
        super(StateEncoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(s, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
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
        return self.model(x)

class InputEncoder(nn.Module):
    def __init__(self, s, hidden_size, output_size, collision_margin=None):
        super(InputEncoder, self).__init__()

        self.output_size = output_size

        # TODO: Not very clear how to do this.
        if collision_margin is None:
            self.model = nn.Sequential(
                nn.Linear(s, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size+1),
            )
        else:
            self.model = nn.Sequential(
                nn.Linear(s, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size),
            )
        self.round = tut.Round()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x, temp=1, n_timesteps = None, collision_margin = None):
        """
        Args:
            x: [n_relations, state_size] (object state)
        Returns:
            [n_relations, output_size]
            [n_relations, selector_size]
        """
        o = self.model(x)
        if collision_margin is None:
            # er, u_logits = o.chunk(2, -1)
            er, u_logits = o[..., :self.output_size], o[..., -1:]
            er = self.relu(er)
            # TODO: implement stochastic.

            # Note: Hard determin.
            u = self.round(self.sigmoid(u_logits/ temp))

            # Note: Soft determin
            # u = self.sigmoid(u_logits / temp)

            # Note: Soft Stochastic
            # if noise and self.count<8000:
            #     u_logits = u_logits + torch.randn(u_logits.shape).to(u_logits.device)*0.1
            #     self.count+=1

            # u_logits = u_logits.tanh() * 8.8
            # u_post = ut.NumericalRelaxedBernoulli(logits=u_logits, temperature=temp)
            # # Unbounded
            # u = u_post.rsample()
            # u = self.round(torch.sigmoid(u / temp))
        else:
            er = o
            u_logits = None
            u = get_u_states_mask(x, collision_margin, n_timesteps)
        # u = self.sigmoid(u_logits / temp)

        # Mask only relevant relations states? At this moment this is done in RelationPropagator
        # er = er * s
        return er, u, u_logits

class InputPropagator(nn.Module):
    def __init__(self, es, ud, output_size, residual=False):
        super(InputPropagator, self).__init__()

        self.residual = residual

        self.linear = nn.Linear(es + ud, output_size)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

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

class Embedding(nn.Module):
    def __init__(self, eo, hidden_size, g):
        super(Embedding, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(eo, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, g),
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
        return self.model(x)

class ComposedEmbedding(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ComposedEmbedding, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            # TODO: ReLU() ?
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
        return self.model(x)

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
        o = torch.clamp(o, min=-1, max=1) # TODO: make sure this works well.

        return o

class EncoderNetwork(nn.Module):
    def __init__(self, input_dim, nf_particle, nf_effect, g_dim, r_dim, u_dim, with_interactions=False, residual=False, collision_margin=None, n_timesteps=None):
        #TODO: added r_dim
        super(EncoderNetwork, self).__init__()

        eo_dim = nf_particle
        es_dim = nf_particle
        self.with_interactions = with_interactions
        self.collision_margin = collision_margin
        self.n_timesteps = n_timesteps

        '''Encoder Networks'''
        self.s_encoder = StateEncoder(input_dim, hidden_size=nf_particle, output_size=es_dim)

        self.u_encoder = InputEncoder(input_dim, hidden_size=nf_effect, output_size=u_dim, collision_margin=collision_margin)
        self.u_propagator = InputPropagator(es_dim, u_dim, output_size=u_dim)

        self.embedding = Embedding(es_dim, hidden_size=nf_particle, g=g_dim)

        input_emb_dim = es_dim + u_dim

        if with_interactions:
            self.rel_encoder = RelationEncoder(input_dim, hidden_size=nf_effect, output_size=r_dim, collision_margin=collision_margin)
            self.rel_propagator = RelationPropagator(es_dim, r_dim, output_size=r_dim)
            input_emb_dim += r_dim

        self.composed_embedding = ComposedEmbedding(input_emb_dim, hidden_size=nf_particle, output_size=g_dim)

        self.round = tut.Round()
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, states, temp, input=None, n_timesteps=None, collision_margin=None):

        # TODO: Initial state need not be the pose.
        #  Check if in simulate T=1
        # ut.print_var_shape(states, 's')
        B, N, T, sd = states.shape
        BT = B*T
        states = states.permute(0, 2, 1, 3) # [B, T, N, sd]
        states = states.reshape(BT, N, sd) # [B * T, N, sd]


        # ut.print_var_shape(rel_states, 'rel_state')
        enc_obj = self.s_encoder(states)
        # ut.print_var_shape(enc_obj, 'enc_obj')
        enc_u, u, u_logits = self.u_encoder(states, temp=temp, n_timesteps=self.n_timesteps, collision_margin=collision_margin)
        # u = get_u_states_mask(states) # TODO: get states of u hardcoded.

        # ut.print_var_shape(enc_u, 'enc_u')
        # ut.print_var_shape(u, 'u')

        enc_obj_receiver = enc_obj[:, :, None, :].repeat(1, 1, N, 1).reshape(BT, N*N, -1)
        enc_obj_sender   = enc_obj[:, None, :, ].repeat(1, N, 1, 1).reshape(BT, N*N, -1)

        # rel_states = (enc_obj_receiver - enc_obj_sender).reshape(BT, N*N, -1) # Note: Absolute value? Square?
        rel_states = (states[:, :, None, :] - states[:, None, :, ]).reshape(BT, N*N, -1)

        obs_u = self.u_propagator(torch.cat([enc_obj, enc_u], dim=-1))

        if input is None:
            obs_u = obs_u #* u
        else:
            # ut.print_var_shape([obs_u, input], 'obs and input')
            obs_u = obs_u * input

        # u_max = torch.clamp(u.sum(-1, keepdims=True), min=0, max=1)
        enc_comp_obj = torch.cat([enc_obj, obs_u], dim=-1)

        if self.with_interactions:
            # TODO: !!!! exchange predictions for objects. o1-o2 --> o2-o1.
            #  This is the same as transposing randomly the matrix of relative obj. And making sel symmetric.
            # TODO: hardcodeinteractions in the rel_encoder. check what are pose distances.
            #  do the same with u? We assume to know limits of the frame.

            # ut.print_var_shape([rel_states, enc_obj_receiver], 'rel states and receiver obj')
            # ut.print_var_shape([enc_rel, sel, sel_logits], 'enc_rel')

            enc_rel, sel, sel_logits = self.rel_encoder(rel_states, temp=temp, n_timesteps=self.n_timesteps, collision_margin=collision_margin)

            # sel = get_rel_states_mask(rel_states) # TODO: get mask for relative states hardcoded.

            obs_rel = self.rel_propagator(torch.cat([enc_obj_receiver, enc_obj_sender, enc_rel], dim=-1))

            # We ignore the diagonal in relations.
            # mask_I = torch.ones([B, N, N, 1]).to(states.device)
            # ids = torch.arange(N)
            # mask_I[:, ids, ids, :] = 0
            mask_I = (- torch.eye(N)[None, :, :, None].to(states.device) + 1.0).reshape(1, N*N, 1)
            # ut.print_var_shape([obs_rel, mask_I, sel], 'before masking')

            obs_rel = (obs_rel * mask_I * sel).reshape(BT, N, N, -1).sum(2)
            # Note: We sum up all interactions in dim 2
            #  so we want to sparsify sel dim=2.

            enc_comp_obj = torch.cat([enc_comp_obj, obs_rel], dim=-1)

            sel_max = torch.clamp((mask_I  * sel).reshape(BT, N, N, -1).sum(2), min=0, max=1) # TODO: Objects in dim -2 are the senders

            # Option: Softmax to chose between activations.
            # ut.print_var_shape([sel_max, u_max], 'selmax umax')
            # masks = torch.cat([u_max, sel_max, torch.clamp(((1-u_max) + (1-sel_max)), min=0, max=1)], dim=-1)
            # masks = self.softmax(masks)

            # obs = self.embedding(enc_obj) * (1-u_max) * (1-sel_max)# Embedding could be based on all.

            sel   = sel  .reshape(B, T, N, N).permute(0, 2, 1, 3).reshape(B, N, T, N) # We show the selector for every object in dim=-2
            if sel_logits is not None:
                sel_logits = sel_logits.reshape(B, T, N, N).permute(0, 2, 1, 3).reshape(B, N, T, N)

            # ut.print_var_shape([sel_max, u_max], 'selmax umax')
        else:
            sel_max = 0
            obs = self.embedding(enc_obj) #* (1-u)# Rounded
            sel, sel_logits = None, None

        # TODO: should I leave the embedding for sure? In case I add to the embedding it should be separately rel and u

        comp_obs = self.composed_embedding(enc_comp_obj) #* u #* self.round(torch.clamp(u_max + sel_max, min=0, max=1))
        # TODO: only composed embedding now. We mask only the object encoding.
        obs = obs #+ comp_obs

        # TODO: Bonissima.
        #  Idea 1: Deixar utilitzar comp_obs nomes x time_steps en training, despres es tot obs. Aixi hi ha backprop
        #  Idea 2: El mateix amb Ag en tots menys 1
        #  Idea 3: El primer step es sempre amb A*obs.

        '''Disclamers'''
        # TODO: Here would originally come another net to unite them all. Particle predictor.
        # Note: Rel Attrs is the relation attributes such as spring constant. In free space we ignore them.
        # TODO: We should add a network that maps noise (or ones) to something. Like gravitational constant.
        #  If we don't, we have to let know that we assume no external interaction but the objects collision to the environment.

        # return (obs, obs_rel, obs_u), (sel, sel_logits), (u, u_logits)

        obs = obs.reshape(B, T, N, -1).permute(0, 2, 1, 3).reshape(B, N, T, -1)
        u   = u  .reshape(B, T, N, -1).permute(0, 2, 1, 3).reshape(B, N, T, -1)
        if u_logits is not None:
            u_logits = u_logits.reshape(B, T, N, -1).permute(0, 2, 1, 3).reshape(B, N, T, -1)

        return obs, (u, u_logits), (sel, sel_logits)

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
                # print(inputs.shape)
                in_t = inputs[..., t, :]
            g, u_tp, sel_tp = self.mapping(in_full_state, temp=temp, collision_margin=self.collision_margin, input=in_t)
            u, u_logits = u_tp
            sel, sel_logits = sel_tp
            if u is None:
                logit_out = True
            # ut.print_var_shape([g, u, u_logits, sel, sel_logits], 'g, us and sels')

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
