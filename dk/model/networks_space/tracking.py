from torch import nn
import torch
from functools import reduce
import numpy as np
import random
from utils import tracker_util as ut

class TrackerArray(nn.Module):

    def __init__(self, n_objects, dims):
        super(TrackerArray, self).__init__()
        self.n_objects = n_objects
        self.dims = dims
        self.ntm = NTM(n_objects, dims)

    def forward(self, h_o_prev, y_e_prev, feats):
        """
        h_o_prev: N * O * dim_h_o
        y_e_prev: N * O * dim_y_e
        C_o_seq:  N * T * C2_1 * C2_2
        """
        bs, T, hw, c = feats.size()

        # Split the input
        C_o = torch.unbind(feats, dim=1) # N * C2_1 * C2_2
        # Iterate over time
        h_o, y_e, y_l, y_p, Y_s, Y_a = {}, {}, {}, {}, {}, {}
        for t in range(0, T):
            self.ntm.t = t
            self.ntm.ntm_cell.t = t
            h_o[t], y_e[t], y_l[t], y_p[t], Y_s[t], Y_a[t] = self.ntm(h_o_prev, y_e_prev, C_o[t], T) # N * O * dim_xx
            h_o_prev, y_e_prev = h_o[t], y_e[t]

        # Merge the states
        h_o_seq = torch.stack(tuple(h_o.values()), dim=1) # N * T * O * dim_h_o
        y_e_seq = torch.stack(tuple(y_e.values()), dim=1) # N * T * O * dim_y_e
        y_l_seq = torch.stack(tuple(y_l.values()), dim=1) # N * T * O * dim_y_l
        y_p_seq = torch.stack(tuple(y_p.values()), dim=1) # N * T * O * dim_y_p
        Y_s_seq = torch.stack(tuple(Y_s.values()), dim=1) # N * T * O * 1 * h * w
        Y_a_seq = torch.stack(tuple(Y_a.values()), dim=1) # N * T * O * D * h * w
        return h_o_seq, y_e_seq, y_l_seq, y_p_seq, Y_s_seq, Y_a_seq


class NTM(nn.Module):

    def __init__(self, n_objects, dims):
        super(NTM, self).__init__()
        self.n_objects = n_objects
        conf_dim, layer_dim, pose_dim, shape_dim, app_dim = dims['confidence'], dims['layer'], dims['pose'], dims['shape'], dims['appearance']
        dim_y = conf_dim + layer_dim + pose_dim + shape_dim + app_dim
        self.dims = dims
        self.dim_h_o = dims['tracker_state']
        # self.fcn = nn.Sequential(nn.Linear(self.dim_h_o, self.dim_h_o),
        #                            nn.ReLU(),
        #                            nn.Linear(self.dim_h_o, dim_y)
        #                          ) # TODO: Check adding nn.Batchnorm1d()
        self.fcn = nn.Linear(self.dim_h_o, dim_y)
        self.softmax = nn.Softmax(dim=1)
        self.st_gumbel_sigmoid = ut.STGumbelSigmoid()
        self.st_gumbel_softmax = ut.STGumbelSoftmax(1)
        self.permutation_matrix_calculator = ut.PermutationMatrixCalculator()
        # dim, feat_dim, n_objects, H, W
        self.ntm_cell = NTMCell(self.dim_h_o, dims['features'], n_objects)
        self.t = 0
        self.att = None
        self.mem = None

    def forward(self, h_o_prev, y_e_prev, C_o, T):
        """
        h_o_prev: N * O * dim_h_o
        y_e_prev: N * O * dim_y_e
        C_o:      N * C2_1 * C2_2
        """
        # o = self.o
        dims = self.dims
        conf_dim, layer_dim, pose_dim, shape_dim, app_dim = dims['confidence'], dims['layer'], dims['pose'], dims['shape'], dims['appearance']

        '''Visualize'''
        # if self.att is None and self.mem is None:
        #     self.att = torch.Tensor(T, self.n_objects, self.ntm_cell.ha, self.ntm_cell.wa).to(h_o_prev.device)
        #     self.mem = torch.Tensor(T, self.n_objects, self.ntm_cell.ha, self.ntm_cell.wa).to(h_o_prev.device)

        '''TEM?'''
        # if "no_tem" in o.exp_config:
        #     h_o_prev = torch.zeros_like(h_o_prev).cuda()
        #     y_e_prev = torch.zeros_like(y_e_prev).cuda()

        # Sort h_o_prev and y_e_prev
        # Note: REP
        # delta = torch.arange(0, self.n_objects).float().to(h_o_prev.device).unsqueeze(0) * 0.0001 # 1 * O
        # y_e_prev_mdf = y_e_prev.squeeze(2).round() - delta
        # perm_mat = self.permutation_matrix_calculator(y_e_prev_mdf) # N * O * O
        # h_o_prev = perm_mat.bmm(h_o_prev) # N * O * dim_h_o
        # y_e_prev = perm_mat.bmm(y_e_prev) # N * O * dim_y_e

        # Update h_o
        h_o_prev_split = torch.unbind(h_o_prev, 1) # N * dim_h_o
        h_o_split = {}
        k_split = {}
        r_split = {}
        for i in range(0, self.n_objects):
            self.ntm_cell.i = i
            # TODO: Maybe add h_o of previous object.
            h_o_split[i], C_o, k_split[i], r_split[i] = self.ntm_cell(h_o_prev_split[i], C_o)
        h_o = torch.stack(tuple(h_o_split.values()), dim=1) # N * O * dim_h_o
        k = torch.stack(tuple(k_split.values()), dim=1) # N * O * C2_2
        r = torch.stack(tuple(r_split.values()), dim=1) # N * O * C2_2
        # att = self.ntm_cell.att
        # mem = self.ntm_cell.mem

        # Recover the original order of h_o
        # Note: REP
        # perm_mat_inv = perm_mat.transpose(1, 2) # N * O * O
        # h_o = perm_mat_inv.bmm(h_o) # N * O * dim_h_o
        # k = perm_mat_inv.bmm(k) # N * O * dim_c_2
        # r = perm_mat_inv.bmm(r) # N * O * dim_c_2

        '''Visualization'''
        # att = perm_mat_inv.data[self.ntm_cell.n].mm(att.view(self.n_objects, -1)).view(self.n_objects, -1, self.ntm_cell.wa) # O * ha * wa
        # mem = perm_mat_inv.data[self.ntm_cell.n].mm(mem.view(self.n_objects, -1)).view(self.n_objects, -1, self.ntm_cell.wa) # O * ha * wa
        # if o.v > 0:
        #     self.att[self.t].copy_(att)
        #     self.mem[self.t].copy_(mem)

        # Generate outputs
        # h_o = smd.CheckBP('h_o', 0)(h_o)
        a = self.fcn(h_o.view(-1, self.dim_h_o)) # NO * dim_y
        a_e = a[:, :conf_dim] # NO * dim_y_e
        a_l = a[:, conf_dim:conf_dim+layer_dim] # NO * dim_y_l
        a_p = a[:, conf_dim+layer_dim:conf_dim+layer_dim+pose_dim] # NO * dim_y_p
        a_s = a[:, conf_dim+layer_dim+pose_dim:conf_dim+layer_dim+pose_dim+shape_dim] # NO * dim_Y_s
        a_a = a[:, conf_dim+layer_dim+pose_dim+shape_dim:] # NO * dim_Y_aa

        # y_e confidence?
        # a_e = smd.CheckBP('a_e', 0)(a_e)
        y_e = a_e#.tanh().abs()
        y_e = y_e.view(-1, self.n_objects, conf_dim) # N * O * dim_y_e

        # y_l layer
        # a_l = smd.CheckBP('a_l', 0)(a_l)
        y_l = self.softmax(a_l)
        ut.norm_grad(y_l, 2)
        y_l = self.st_gumbel_softmax(y_l)
        y_l = y_l.view(-1, self.n_objects, layer_dim) # N * O * dim_y_l

        # y_p pose
        # a_p = smd.CheckBP('a_p', 0)(a_p)
        y_p = a_p#.tanh()
        # y_p = a_p
        y_p = y_p.view(-1, self.n_objects, pose_dim) # N * O * dim_y_p

        # Y_s shape
        # a_s = smd.CheckBP('a_s', 0)(a_s)
        Y_s = a_s.sigmoid()
        Y_s = self.st_gumbel_sigmoid(Y_s)
        Y_s = Y_s.view(-1, self.n_objects, shape_dim) # N * O * 1 * h * w TODO: o.

        # Y_a appearance
        # a_a = smd.CheckBP('a_a', 0)(a_a)
        Y_a = a_a#.sigmoid()
        Y_a = Y_a.view(-1, self.n_objects, app_dim) # N * O * D * h * w TODO: o.

        # adaptive computation time
        '''Adaptive computation'''
        # y_e_perm = perm_mat.bmm(y_e).round() # N * O * dim_y_e
        # y_e_mask = y_e_prev.round() + y_e_perm  # N * O * dim_y_e
        # y_e_mask = y_e_mask.lt(0.5).type_as(y_e_mask)
        # y_e_mask = y_e_mask.cumsum(1)
        # y_e_mask = y_e_mask.lt(0.5).type_as(y_e_mask)
        # ones = torch.ones(y_e_mask.size(0), 1, conf_dim).cuda()  # N * 1 * dim_y_e
        # y_e_mask = torch.cat((ones, y_e_mask[:, 0:self.n_objects-1]), dim=1)
        # y_e_mask = perm_mat_inv.bmm(y_e_mask)  # N * O * dim_y_e
        # h_o = y_e_mask * (h_o - h_o_prev) + h_o_prev  # N * O * dim_h_o
        # # h_o = y_e_mask * h_o  # N * O * dim_h_o
        # y_e = y_e_mask * y_e  # N * O * dim_y_e
        # y_p = y_e_mask * y_p  # N * O * dim_y_p
        # Y_a = y_e_mask.view(-1, self.n_objects, conf_dim, 1, 1) * Y_a  # N * O * D * h * w (o.dim_y_e)

        # if self.t == T - 1:
        #     print(y_e.data.view(-1, self.n_objects)[0:1, 0:min(self.n_objects, 10)])

        return h_o, y_e, y_l, y_p, Y_s, Y_a


class NTMCell(nn.Module):

    def __init__(self, dim_h_o, feat_dim, n_objects, H=None, W=None):
        super(NTMCell, self).__init__()
        # self.o = o
        # o.dim_h_o = dim
        # o.dim_C2_2 = feat_dim
        # o.O = n_objects

        self.linear_k = nn.Linear(dim_h_o, feat_dim)
        self.linear_b = nn.Linear(dim_h_o, 1)
        self.linear_e = nn.Linear(dim_h_o, feat_dim)
        self.linear_v = nn.Linear(dim_h_o, feat_dim)
        self.cosine_similarity = nn.CosineSimilarity(dim=2) # Check dimension
        self.softmax = nn.Softmax(dim=1)
        self.rnn_cell = nn.GRUCell(feat_dim, dim_h_o)
        for m in self.modules():
            if isinstance(m, nn.GRUCell):
                nn.init.xavier_normal_(m.weight_ih)
                nn.init.xavier_normal_(m.weight_hh)
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
        # self.ha = int(round(np.sqrt(feat_dim*H/W)))
        # self.wa = int(round(np.sqrt(feat_dim*W/H)))
        # self.att = torch.Tensor(n_objects, self.ha, self.wa)#.cuda()
        # self.mem = torch.Tensor(n_objects, self.ha, self.wa)#.cuda()
        self.i = 0 # object id
        self.t = 0 # time
        self.n = 0 # sample id

    def forward(self, h_o_prev, C):
        """
        h_o_prev: N * dim_h_o
        C: N * C2_1 * C2_2
        """

        '''Visualize'''
        # if o.v > 0:
        #     if self.i == 0:
        #         self.att.fill_(0.5)
        #         self.mem.fill_(0.5)
        #     self.mem[self.i].copy_(C.data[n].mean(1).view(self.ha, self.wa))

        '''Attention'''
        # Addressing key
        # print(h_o_prev.device)

        k = self.linear_k(h_o_prev) # N * C2_2

        if k.shape[0] != C.shape[0]:
            print(k.shape, C.shape)
            k = k.repeat_interleave(C.shape[0]//k.shape[0], dim=0)
        k_expand = k.unsqueeze(1).expand_as(C) # N * C2_1 * C2_2
        # Key strength, which equals to beta_pre.exp().log1p() + 1 but avoids 'inf' caused by exp()
        beta_pre = self.linear_b(h_o_prev)
        beta_pos = beta_pre.clamp(min=0)
        beta_neg = beta_pre.clamp(max=0)
        beta = beta_neg.exp().log1p() + beta_pos + (-beta_pos).exp().log1p() + (1 - np.log(2)) # N * 1
        # Weighting
        C_cos = ut.Identity()(C)
        ut.norm_grad(C_cos, 1)
        s = self.cosine_similarity(C_cos, k_expand).view(-1, C.shape[1]) # N * C2_1
        w = self.softmax(s * beta) # N * C2_1

        '''Read Memory'''
        # Read vector
        w1 = w.unsqueeze(1) # N * 1 * C2_1
        ut.norm_grad(w1, 1)
        r = w1.bmm(C).squeeze(1) # N * C2_2
        # RNN
        h_o = self.rnn_cell(r, h_o_prev)

        '''Write Memory'''
        # Erase vector
        e = self.linear_e(h_o).sigmoid().unsqueeze(1) # N * 1 * C2_2
        # Write vector
        v = self.linear_v(h_o).unsqueeze(1) # N * 1 * C2_2
        # Update memory
        w2 = w.unsqueeze(2) # N * C2_1 * 1
        C = C * (1 - w2.bmm(e)) + w2.bmm(v) # N * C2_1 * C2_2

        '''Visualize'''
        # if o.v > 0:
        #     self.att[self.i].copy_(w.data[n].view(self.ha, self.wa))

        # # C in 3D shape
        # o.dim_C3_1 = o.cnn['out_sizes'][-1][0]
        # o.dim_C3_2 = o.cnn['out_sizes'][-1][1]
        # o.dim_C3_3 = o.cnn['conv_features'][-1]
        # o.dim_h_o = o.dim_C3_3 * 4
        # # C in 2D shape
        # if "no_att" in o.exp_config:
        #     o.dim_C2_1 = 1
        #     o.dim_C2_2 = o.dim_C3_1 * o.dim_C3_2 * o.dim_C3_3
        # else:
        #     o.dim_C2_1 = o.dim_C3_1 * o.dim_C3_2
        #     o.dim_C2_2 = o.dim_C3_3
        # if "no_occ" in o.exp_config:
        #     o.dim_y_l = 1
        # o.dim_y_e = 1
        # o.dim_y_p = 4
        # o.dim_Y_s = 1 * o.h * o.w
        # o.dim_Y_a = o.D * o.h * o.w
        return h_o, C, k, r