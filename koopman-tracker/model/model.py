import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from base import BaseModel

from model.networks_space import ImageEncoder, ImageDecoder, ImageBroadcastDecoder, KoopmanOperators
from model.networks_space.spatial_tf import SpatialTransformation
from torch.distributions import Normal, kl_divergence
from functools import reduce
from torch.autograd import Variable
from torch.distributions.categorical import Categorical
import random
import matplotlib.pyplot as plt

def _get_flat(x, keep_dim=False):
    if keep_dim:
        return x.reshape(torch.Size([1, x.size(0) * x.size(1)]) + x.size()[2:])
    return x.reshape(torch.Size([x.size(0) * x.size(1)]) + x.size()[2:])

class RecKoopmanModel(BaseModel):
    def __init__(self, in_channels, feat_dim, nf_particle, nf_effect, g_dim, u_dim,
                 n_objects, free_pred = 1, I_factor=10, n_blocks=1, psteps=1, n_timesteps=1, ngf=8, image_size=[64, 64]):
        super().__init__()
        out_channels = 1
        n_layers = int(np.log2(image_size[0])) - 1

        self.u_dim = u_dim

        # Set state dim with config, depending on how many time-steps we want to take into account
        self.image_size = image_size
        self.n_timesteps = n_timesteps
        self.state_dim = feat_dim
        self.I_factor = I_factor
        self.psteps = psteps
        self.g_dim = g_dim
        self.free_pred = free_pred
        self.compositional = True

        feat_dyn_dim = feat_dim // 8
        feat_dyn_dim = 4
        self.feat_dyn_dim = feat_dyn_dim
        self.feat_cte_dim = feat_dim - feat_dyn_dim

        self.with_u = True
        self.n_iters = 1
        self.ini_alpha = 1
        # Note:
        #  - I leave it to 0 now. If it increases too fast, the gradients might be affected
        self.incr_alpha = 0.1


        self.cte_resolution = (32, 32)
        self.ori_resolution = (128, 128)
        self.att_resolution = (16, 16)
        self.obj_resolution = (4, 4)
        # self.n_objects = reduce((lambda x, y: x * y), self.obj_resolution)
        self.n_objects = n_objects

        self.spatial_tf = SpatialTransformation(self.cte_resolution, self.ori_resolution)

        self.linear_f_cte_post = nn.Linear(2 * self.feat_cte_dim, 2 * self.feat_cte_dim)
        self.linear_f_dyn_post = nn.Linear(2 * self.feat_dyn_dim, 2 * self.feat_dyn_dim)
        #
        self.bc_decoder = False                                                    #2 *
        if self.bc_decoder:
            self.image_decoder = ImageBroadcastDecoder(self.feat_cte_dim, out_channels, resolution=(16, 16)) # resolution=self.att_resolution
        else:
            self.image_decoder = ImageDecoder(self.feat_cte_dim, out_channels, dyn_dim = self.feat_dyn_dim)
        self.image_encoder = ImageEncoder(in_channels, 2 * self.feat_cte_dim, self.feat_dyn_dim, self.att_resolution, self.n_objects, ngf, n_layers)  # feat_dim * 2 if sample here
        self.koopman = KoopmanOperators(feat_dyn_dim, nf_particle, nf_effect, g_dim, u_dim, n_timesteps, n_blocks)

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
        # Note: Add annealing in SPACE
        bs, T, ch, h, w = input.shape
        output = {}

        free_pred = self.free_pred #Try again with trained.
        returned_post = []
        f_bb = input

        # Dynamic features
        T_inp = T
        f_dyn, shape, f_cte, confi = self.image_encoder(f_bb[:, :T_inp], block='dyn_track')
        f_dyn = f_dyn.reshape(-1, f_dyn.shape[-1])

        # Sample dynamic features or reshape
        # Option 1: Don't sample
        f_dyn = f_dyn.reshape(bs * self.n_objects, T_inp, -1)
        # Option 2: Sample
        # f_mu_dyn, f_logvar_dyn = self.linear_f_dyn_post(f_dyn).reshape(bs, self.n_objects, T_inp, -1).chunk(2, -1)
        # f_dyn_post = Normal(f_mu_dyn, F.softplus(f_logvar_dyn))
        # f_dyn = f_dyn_post.rsample()
        # f_dyn = f_dyn.reshape(bs * self.n_objects, T_inp, -1)
        # returned_post.append(f_dyn_post)

        # Get delayed dynamic features
        f_dyn_state, T_inp = self._get_full_state_hankel(f_dyn, T_inp)

        # Get inputs from delayed dynamic features
        if not self.with_u:
            u, u_post, u_logit = self.koopman.to_u(f_dyn_state, temp=self.ini_alpha + epoch * self.incr_alpha, ignore=True)
            u = torch.zeros_like(u)
        else: u, u_post, u_logit = self.koopman.to_u(f_dyn_state, temp=self.ini_alpha + epoch * self.incr_alpha, ignore=False)

        # Get observations from delayed dynamic features # TODO: We could map this to double of g. The other half being a one-hot vector. No need to sparsify

        if not self.compositional:
            g = self.koopman.to_g(f_dyn_state.reshape(bs * self.n_objects * T_inp, -1), self.psteps)
            g = g.reshape(bs * self.n_objects, T_inp, *g.shape[1:])
        else:
            g, g_mask = self.koopman.to_composed_g(f_dyn_state.reshape(bs * self.n_objects * T_inp, -1), self.psteps)
            g = g.reshape(bs * self.n_objects, T_inp, *g.shape[1:])
            g_mask = g_mask.reshape(bs * self.n_objects, T_inp, *g_mask.shape[1:])[:, :1]
            # g_mask = g_mask.reshape(bs * self.n_objects, T_inp, *g_mask.shape[1:])[:, :-1]
            #TODO: same for all time-steps? or different at each time step?

        # Get shifted observations for sys ID
        randperm = torch.arange(g.shape[0]) # No permutation
        # randperm = torch.randperm(g.shape[0]) # No permutation

        if not self.compositional:
            g_mask = torch.ones_like(g[:, :1])

        if free_pred > 0:
            G_tilde = g[randperm, :-1-free_pred, None] * g_mask
            H_tilde = g[randperm, 1:-free_pred, None] * g_mask
        else:
            G_tilde = g[randperm, :-1, None] * g_mask
            H_tilde = g[randperm, 1:, None] * g_mask

        # Sys ID
        A, B, A_inv, fit_err = self.koopman.system_identify(G=G_tilde, H=H_tilde, U=u[randperm, :T_inp-free_pred-1], I_factor=self.I_factor) # Try not permuting U when inp is permutted

        # Rollout from start_step onwards.
        start_step = 0 # g and u must be aligned!!
        # TODO: Re-check: Why a constant g is valid. Unless t-1 contains time t. pero aun asi no se yo. ITS MAKING THE BOX BIGGER! CONSTRAIN AFTER KOOPMAN.
        G_for_pred = self.koopman.simulate(T=T_inp-start_step-1, g=g[:,start_step], u=u[:,start_step:], A=A, B=B)
        g_for_koop= G_for_pred

        # TODO: predict pose and confi with logits
        rec = {"obs": g,
               "confi": confi[:, :, self.n_timesteps-1:self.n_timesteps-1 + T_inp], # TODO: Set confi to 1 after some epochs (increase temp of sigmoid)
               "shape": shape[:, :, self.n_timesteps-1:self.n_timesteps-1 + T_inp],
               "f_cte": f_cte[:, :, self.n_timesteps-1:self.n_timesteps-1 + T_inp],
               "T": T_inp,
               "name": "rec"}
        pred = {"obs": G_for_pred,
                "confi": confi[:, :, -G_for_pred.shape[1]:],
                "shape": shape[:, :, -G_for_pred.shape[1]:],
                "f_cte": f_cte[:, :, -G_for_pred.shape[1]:],
                "T": G_for_pred.shape[1],
                "name": "pred"}
        outs = {}

        # TODO: Check if the indices for f_bb and supervision are correct.
        # Recover partial shape with decoded dynamical features. Iterate with new estimates of the appearance.
        # Note: This process could be iterative.
        for idx, case in enumerate([rec, pred]):

            case_name = case["name"]

            f_dyn_state = self.koopman.to_s(gcodes=_get_flat(case["obs"]),
                                          psteps=self.psteps)

            # Encode with raw pose and confidence features
            pose = f_dyn_state.tanh()
            # confi = f_dyn_state[..., :confi.shape[-1]].reshape(bs * self.n_objects * case["T"], -1).tanh().abs()

            # appearance features
            f_cte = case["f_cte"].reshape(bs * self.n_objects * case["T"], -1)
            confi = case["confi"].reshape(bs * self.n_objects * case["T"], -1).tanh().abs()

            # Sample appearance (we call appearance features f_cte)
            f_mu_cte, f_logvar_cte = self.linear_f_cte_post(f_cte).reshape(bs, self.n_objects, case["T"], -1).chunk(2, -1)
            f_cte_post = Normal(f_mu_cte, F.softplus(f_logvar_cte))
            f_cte = f_cte_post.rsample().reshape(bs * self.n_objects * case["T"], -1)

            # Register statistics
            returned_post.append(f_cte_post)

            # Get full feature vector
            f = f_cte

            # Get output with decoder
            dec_obj = self.image_decoder(f, block='to_x')

            if not self.bc_decoder:
                grid, area = self.spatial_tf(confi, pose)
                outs[case_name], out_shape = self.spatial_tf.warp_and_render(dec_obj, case["shape"], confi, grid)
            else:
                outs[case_name] = dec_obj * confi[..., None, None]

        out_rec = outs["rec"]
        out_pred = outs["pred"]

        ''' Returned variables '''
        returned_g = torch.cat([g, G_for_pred], dim=1) # Observations

        output["rec"] = torch.clamp(torch.sum(out_rec.reshape(bs, self.n_objects, -1, *out_rec.size()[1:]), dim=1), min=0, max=1)
        output["pred"] = torch.clamp(torch.sum(out_pred.reshape(bs, self.n_objects, -1, *out_rec.size()[1:]), dim=1), min=0, max=1)
        output["obs_rec_pred"] = returned_g.reshape(bs * self.n_objects, -1, returned_g.shape[-1])
        output["gauss_post"] = returned_post
        output["A"] = A
        output["B"] = B
        output["u"] = u.reshape(bs * self.n_objects, -1, u.shape[-1])
        output["u_bern_logit"] = u_logit.reshape(bs, self.n_objects, -1, u_logit.shape[-1]) # Input distribution

        # o_touple = (out_rec, out_pred, returned_g.reshape(torch.Size([bs * self.n_objects, -1]) + returned_g.size()[-1:]))
        # o = [item.reshape(torch.Size([bs * self.n_objects, -1]) + item.size()[1:]) for item in o_touple]

        # Option 1: one object mapped to 0
        # shape_o0 = o[0].shape
        # o[0] = o[0].reshape(bs, self.n_objects, *o[0].shape[1:])
        # o[0][:,0] = o[0][:,0]*0
        # o[0] = o[0].reshape(*shape_o0)

        # Sum across objects. Note: Add Clamp or Sigmoid.
        # o[:2] = [torch.clamp(torch.sum(item.reshape(bs, self.n_objects, *item.shape[1:]), dim=1), min=0, max=1) for item in o[:2]]

        # TODO: output = {}
        #  output['rec'] = o[0]
        #  output['pred'] = o[1]

        # o.append(returned_post)
        # o.append(A) # State transition matrix
        # o.append(B) # Input matrix
        # o.append(u.reshape(bs * self.n_objects, -1, u.shape[-1])) # Inputs
        # o.append(u_logit.reshape(bs, self.n_objects, -1, u_logit.shape[-1])) # Input distribution
        # o.append(g_for_koop.reshape(bs * self.n_objects, -1, g_for_koop.shape[-1])) # Observation propagated only with A
        # o.append(fit_err) # Fit error g to G_for_pred

        return output