from abc import ABC

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

# from data import denormalize, normalize
# from utils import load_data


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


class PropagationNetwork(nn.Module):

    def __init__(self, input_particle_dim, nf_particle, nf_effect, output_dim,
                 tanh=False, residual=False, use_gpu=False):

        super(PropagationNetwork, self).__init__()

        self.use_gpu = use_gpu
        self.residual = residual

        # (1) state
        self.obj_encoder = ParticleEncoder(input_particle_dim, nf_particle, nf_effect)

        # Note: when we get to the point of encoding relations, we are making a (strong) assumption that a single object doesn't have interacting dynamics


        # (1) state receiver (2) state_diff
        # self.relation_encoder = RelationEncoder(input_relation_dim, nf_relation, nf_relation)

        # (1) relation encode (2) sender effect (3) receiver effect
        # self.relation_propagator = Propagator(nf_relation + 2 * nf_effect, nf_effect)

        # (1) particle encode (2) particle effect
        # self.particle_propagator = Propagator(2 * nf_effect, nf_effect, self.residual)
        self.particle_propagator = Propagator(nf_effect, nf_effect, self.residual)

        # rigid predictor
        # (1) particle encode (2) set particle effect
        self.particle_predictor = ParticlePredictor(nf_effect, nf_effect, output_dim)

        if tanh:
            self.particle_predictor = nn.Sequential(
                self.particle_predictor, nn.Tanh()
            )

    def forward(self, states, pstep):
        """
        :param states: B x N x state_dim
        :param pstep: 1 or 2
        :return:
        """
        # Note: Sizes for Rope
        # torch.Size([8, 64, 6, 4])

        '''encode node'''
        # Note: attrs indicate meta-information about the state. Which kind of node are we talking about, for instance
        obj_encode = self.obj_encoder(states)


        '''encode edge'''
        # rel_states = states[:, :, None, :] - states[:, None, :, :]
        # receiver_attr = attrs[:, :, None, :].repeat(1, 1, N, 1)
        # sender_attr = attrs[:, None, :, :].repeat(1, N, 1, 1)
        # tmp = torch.cat([rel_attrs, rel_states, receiver_attr, sender_attr], 3)
        # rel_encode = self.relation_encoder(tmp.reshape(B * N * N, -1)).reshape(B, N, N, -1)

        for i in range(pstep):
            '''calculate relation effect'''
            # receiver_code = obj_encode[:, :, None, :].repeat(1, 1, N, 1)
            # sender_code = obj_encode[:, None, :, :].repeat(1, N, 1, 1)
            # tmp = torch.cat([rel_encode, receiver_code, sender_code], 3)
            # rel_effect = self.relation_propagator(tmp.reshape(B * N * N, -1)).reshape(B, N, N, -1)

            '''aggregating relation effect'''
            # rel_agg_effect = rel_effect.sum(2)

            '''calc particle effect'''
            # tmp = torch.cat([obj_encode, rel_agg_effect], 2)
            tmp = obj_encode
            obj_encode = self.particle_propagator(tmp)

        obj_prediction = self.particle_predictor(obj_encode)
        return obj_prediction


# ======================================================================================================================
class KoopmanOperators(nn.Module, ABC):
    def __init__(self, state_dim, nf_particle, nf_effect, g_dim, n_timesteps, residual=False):
        super(KoopmanOperators, self).__init__()

        # self.stat = load_data(['attrs', 'states', 'actions'], args.stat_path)

        self.residual = residual

        ''' state '''
        # Note: True? we should not include action in state encoder
        input_particle_dim = state_dim * n_timesteps

        self.mapping = PropagationNetwork(
            input_particle_dim=input_particle_dim, nf_particle=nf_particle,
            nf_effect=nf_effect, output_dim=g_dim * 2, tanh=False,  # use tanh to enforce the shape of the code space
            residual=residual)

        # the state for decoding phase is replaced with code of g_dim
        input_particle_dim = g_dim

        # print('state_decoder', 'node', input_particle_dim, 'edge', input_relation_dim)

        self.inv_mapping = PropagationNetwork(
            input_particle_dim=input_particle_dim, nf_particle=nf_particle,
            nf_effect=nf_effect, output_dim=state_dim, tanh=False, residual=residual)

        ''' dynamical system coefficient: A'''
        self.A = None

        # self.system_identify = self.fit
        # self.simulate = self.rollout
        # self.step = self.linear_forward

        self.system_identify = self.fit_diagonal
        self.simulate = self.rollout_diagonal
        self.step = self.linear_forward_diagonal

    def to_s(self, gcodes, pstep):
        """ state decoder """
        states = self.inv_mapping(states=gcodes, pstep=pstep)
        # regularize_state_Soft(states, rel_attrs, self.stat)
        return states

    def to_g(self, states, pstep):
        """ state encoder """
        return self.mapping(states=states, pstep=pstep)

    @staticmethod
    def get_aug(G, rel_attrs):
        """
        :param G: B x T x N x D
        :param rel_attrs:  B x N x N x R
        :return:
        """
        B, T, N, D = G.size()
        R = rel_attrs.size(-1)

        sumG_list = []
        for i in range(R):
            ''' B x T x N x N '''
            adj = rel_attrs[:, :, :, i][:, None, :, :].repeat(1, T, 1, 1)
            sumG = torch.bmm(
                adj.reshape(B * T, N, N),
                G.reshape(B * T, N, D)
            ).reshape(B, T, N, D)
            sumG_list.append(sumG)

        augG = torch.cat(sumG_list, 3)

        return augG

    # structured A

    def fit(self, G, H, I_factor=10):
        """
        :param G: B x T x N x D
        :param H: B x T x N x D
        :param rel_attrs: B x N x N x R (relation_dim) rel_attrs[i,j] ==> receiver i, sender j
        :param I_factor: scalor
        :return:
        A: B x R D x D
        s.t.
        H = augG @ A
        """

        ''' B x R: sqrt(# of appearance of block matrices of the same type)'''
        # rel_weights = torch.sqrt(rel_attrs.sum(1).sum(1))
        # rel_weights = torch.clamp(rel_weights, min=1e-8)

        bs, T, N, D = G.size()
        # R = rel_attrs.size(-1)

        ''' B x T x N x R D '''
        # augG = self.get_aug(G, rel_attrs)
        # augG_reweight = augG.reshape(bs, T, N, R, D) / rel_weights[:, None, None, :, None]

        ''' B x TN x RD'''
        # G_reweight = augG_reweight.reshape(bs, T * N, R * D)
        G_reweight = G.reshape(bs, T * N, D)

        '''B x (R * D) x D'''
        A_reweight = torch.bmm(
            self.batch_pinv(G_reweight, I_factor),
            H.reshape(bs, T * N, D)
        )
        # self.A = A_reweight[:, :R * D].reshape(bs, R, D, D) / rel_weights[:, :, None, None]
        # self.A = self.A.reshape(bs, R * D, D)
        self.A = A_reweight.reshape(bs, D, D)

        fit_err = H.reshape(bs, T * N, D) - torch.bmm(G_reweight, A_reweight)
        fit_err = torch.sqrt((fit_err ** 2).mean())

        return self.A, fit_err

    def linear_forward(self, g, A):
        """
        :param g: B x N x D
        :param u: B x N x a_dim
        :param rel_attrs: B x N x N x R
        :return:
        """
        ''' B x N x R D '''
        # aug_g = self.get_aug(G=g[:, None, :, :], rel_attrs=rel_attrs)[:, 0]
        aug_g = g[:, None, :]

        new_g = torch.bmm(aug_g, A)
        return new_g

    def rollout(self, g, T, A):
        """
        :param g: B x N x D
        :param rel_attrs: B x N x N x R
        :param T:
        :return:
        """
        g_list = []
        for t in range(T):
            g = self.linear_forward(g, A)[:,0] # For single object
            g_list.append(g[:, None, :])
        return torch.cat(g_list, 1)

#####################################################################################
    def fit_diagonal(self, G, H, I_factor):
        bs, T, N, D = G.size()
        G, H = G.permute(0, 3, 1, 2), H.permute(0, 3, 1, 2)

        '''B x (D) x D'''
        A = torch.bmm(
            self.batch_pinv(G.reshape(bs * D, T * N, 1), I_factor),
            H.reshape(bs * D, T * N, 1)
        )
        # self.A = A

        fit_err = H.reshape(bs * D, T * N, 1) - torch.bmm(G.reshape(bs * D, T * N, 1), A)
        fit_err = torch.sqrt((fit_err ** 2).mean())

        return A, fit_err

    # def linear_forward_diagonal(self, g, A):
    #     new_g = torch.bmm(g, A)
    #     return new_g
    #
    # def rollout_diagonal(self, g, T, A):
    #     g_list = []
    #     for t in range(T):
    #         g = self.linear_forward_diagonal(g, A)
    #         g_list.append(g[:, None, :])
    #     return torch.cat(g_list, 1)

    def linear_forward_diagonal(self, g, A):
        """
        :param g: B x N x D
        :param u: B x N x a_dim
        :param rel_attrs: B x N x N x R
        :return:
        """
        ''' B x N x R D '''
        bs, dim = g.shape
        aug_g = g.reshape(-1, 1, 1)
        new_g = torch.bmm(aug_g, A).reshape(bs, 1, dim)

        return new_g

    def rollout_diagonal(self, g, T, A):
        """
        :param g: B x N x D
        :param rel_attrs: B x N x N x R
        :param T:
        :return:
        """
        g_list = []
        for t in range(T):
            g = self.linear_forward_diagonal(g, A)[:,0] # For single object
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

    # unstructured large A
    #
    # def fit_unstructured(self, G, H, U, I_factor, rel_attrs=None):
    #     """
    #     :param G: B x T x N x D
    #     :param H: B x T x N x D
    #     :param U: B x T x N x a_dim
    #     :param I_factor: scalor
    #     :return: A, B
    #     s.t.
    #     H = catG @ A + catU @ B
    #     """
    #     bs, T, N, D = G.size()
    #     G = G.reshape(bs, T, -1)
    #     H = H.reshape(bs, T, -1)
    #     U = U.reshape(bs, T, -1)
    #
    #     G_U = torch.cat([G, U], 2)
    #     A_B = torch.bmm(
    #         self.batch_pinv(G_U, I_factor),
    #         H
    #     )
    #     self.A = A_B[:, :N * D]
    #     self.B = A_B[:, N * D:]
    #
    #     fit_err = H - torch.bmm(G_U, A_B)
    #     fit_err = torch.sqrt((fit_err ** 2).mean())
    #
    #     return self.A, self.B, fit_err
    #
    # def linear_forward_unstructured(self, g, u, rel_attrs=None):
    #     B, N, D = g.size()
    #     a_dim = u.size(-1)
    #     g = g.reshape(B, 1, N * D)
    #     u = u.reshape(B, 1, N * a_dim)
    #     new_g = torch.bmm(g, self.A) + torch.bmm(u, self.B)
    #     return new_g.reshape(B, N, D)
    #
    # def rollout_unstructured(self, g, u_seq, T, rel_attrs=None):
    #     g_list = []
    #     for t in range(T):
    #         g = self.linear_forward_unstructured(g, u_seq[:, t])
    #         g_list.append(g[:, None, :, :])
    #     return torch.cat(g_list, 1)
    #
    # # shared small A
    #
    # def fit_diagonal(self, G, H, U, I_factor, rel_attrs=None):
    #     bs, T, N, D = G.size()
    #     a_dim = U.size(3)
    #
    #     G_U = torch.cat([G, U], 3)
    #
    #     '''B x (D + a_dim) x D'''
    #     A_B = torch.bmm(
    #         self.batch_pinv(G_U.reshape(bs, T * N, D + a_dim), I_factor),
    #         H.reshape(bs, T * N, D)
    #     )
    #     self.A = A_B[:, :D]
    #     self.B = A_B[:, D:]
    #
    #     fit_err = H.reshape(bs, T * N, D) - torch.bmm(G_U.reshape(bs, T * N, D + a_dim), A_B)
    #     fit_err = torch.sqrt((fit_err ** 2).mean())
    #
    #     return self.A, self.B, fit_err
    #
    # def linear_forward_diagonal(self, g, u, rel_attrs=None):
    #     new_g = torch.bmm(g, self.A) + torch.bmm(u, self.B)
    #     return new_g
    #
    # def rollout_diagonal(self, g, u_seq, T, rel_attrs=None):
    #     g_list = []
    #     for t in range(T):
    #         g = self.linear_forward_diagonal(g, u_seq[:, t])
    #         g_list.append(g[:, None, :, :])
    #     return torch.cat(g_list, 1)

def regularize_state_Soft(states, rel_attrs, stat):
    """
    :param states: B x N x state_dim
    :param rel_attrs: B x N x N x relation_dim
    :param stat: [xxx]
    :return new states: B x N x state_dim
    """
    states_denorm = denormalize([states], [stat[1]], var=True)[0]
    states_denorm_acc = denormalize([states.clone()], [stat[1]], var=True)[0]

    rel_attrs = rel_attrs[0]

    rel_attrs_np = rel_attrs.detach().cpu().numpy()

    def get_rel_id(x):
        return np.where(x > 0)[0][0]

    B, N, state_dim = states.size()
    count = Variable(torch.FloatTensor(np.zeros((1, N, 1, 8))).to(states.device))

    for i in range(N):
        for j in range(N):

            if i == j:
                assert get_rel_id(rel_attrs_np[i, j]) % 9 == 0  # rel_attrs[i, j, 0] == 1
                count[:, i, :, :] += 1
                continue

            assert torch.sum(rel_attrs[i, j]) <= 1

            if torch.sum(rel_attrs[i, j]) == 0:
                continue

            if get_rel_id(rel_attrs_np[i, j]) % 9 == 1:  # rel_attrs[i, j, 1] == 1:
                assert get_rel_id(rel_attrs_np[j, i]) % 9 == 2  # rel_attrs[j, i, 2] == 1
                x0 = 1;
                y0 = 3
                x1 = 0;
                y1 = 2
                idx = 1
            elif get_rel_id(rel_attrs_np[i, j]) % 9 == 2:  # rel_attrs[i, j, 2] == 1:
                assert get_rel_id(rel_attrs_np[j, i]) % 9 == 1  # rel_attrs[j, i, 1] == 1
                x0 = 3;
                y0 = 1
                x1 = 2;
                y1 = 0
                idx = 2
            elif get_rel_id(rel_attrs_np[i, j]) % 9 == 3:  # rel_attrs[i, j, 3] == 1:
                assert get_rel_id(rel_attrs_np[j, i]) % 9 == 4  # rel_attrs[j, i, 4] == 1
                x0 = 0;
                y0 = 1
                x1 = 2;
                y1 = 3
                idx = 3
            elif get_rel_id(rel_attrs_np[i, j]) % 9 == 4:  # rel_attrs[i, j, 4] == 1:
                assert get_rel_id(rel_attrs_np[j, i]) % 9 == 3  # rel_attrs[j, i, 3] == 1
                x0 = 1;
                y0 = 0
                x1 = 3;
                y1 = 2
                idx = 4
            elif get_rel_id(rel_attrs_np[i, j]) % 9 == 5:  # rel_attrs[i, j, 5] == 1:
                assert get_rel_id(rel_attrs_np[j, i]) % 9 == 8  # rel_attrs[j, i, 8] == 1
                x = 0;
                y = 3
                idx = 5
            elif get_rel_id(rel_attrs_np[i, j]) % 9 == 8:  # rel_attrs[i, j, 8] == 1:
                assert get_rel_id(rel_attrs_np[j, i]) % 9 == 5  # rel_attrs[j, i, 5] == 1
                x = 3;
                y = 0
                idx = 8
            elif get_rel_id(rel_attrs_np[i, j]) % 9 == 6:  # rel_attrs[i, j, 6] == 1:
                assert get_rel_id(rel_attrs_np[j, i]) % 9 == 7  # rel_attrs[j, i, 7] == 1
                x = 1;
                y = 2
                idx = 6
            elif get_rel_id(rel_attrs_np[i, j]) % 9 == 7:  # rel_attrs[i, j, 7] == 1:
                assert get_rel_id(rel_attrs_np[j, i]) % 9 == 6  # rel_attrs[j, i, 6] == 1
                x = 2;
                y = 1
                idx = 7
            else:
                AssertionError("Unknown rel_attr %f" % rel_attrs[i, j])

            if idx < 5:
                # if connect by two points
                x0 *= 2;
                y0 *= 2
                x1 *= 2;
                y1 *= 2
                count[:, i, :, x0:x0 + 2] += 1
                count[:, i, :, x1:x1 + 2] += 1
                states_denorm_acc[:, i, x0:x0 + 2] += states_denorm[:, j, y0:y0 + 2]
                states_denorm_acc[:, i, x0 + 8:x0 + 10] += states_denorm[:, j, y0 + 8:y0 + 10]
                states_denorm_acc[:, i, x1:x1 + 2] += states_denorm[:, j, y1:y1 + 2]
                states_denorm_acc[:, i, x1 + 8:x1 + 10] += states_denorm[:, j, y1 + 8:y1 + 10]

            else:
                # if connected by a corner
                x *= 2;
                y *= 2
                count[:, i, :, x:x + 2] += 1
                states_denorm_acc[:, i, x:x + 2] += states_denorm[:, j, y:y + 2]
                states_denorm_acc[:, i, x + 8:x + 10] += states_denorm[:, j, y + 8:y + 10]

    states_denorm = states_denorm_acc.view(B, N, 2, state_dim // 2) / count
    states_denorm = states_denorm.view(B, N, state_dim)

    return normalize([states_denorm], [stat[1]], var=True)[0]