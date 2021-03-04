import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from base import BaseModel

from model.networks_space import ImageEncoder, ImageDecoder, ImageBroadcastDecoder
from model.networks_space.consistent_iterative_koopman import KoopmanOperators
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
                 n_objects, free_pred = 1, I_factor=10, n_blocks=1, psteps=1, n_timesteps=1, ngf=8, image_size=[64, 64], num_sys=0, batch_size=40):
        super().__init__()
        out_channels = in_channels
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

        # feat_dyn_dim = feat_dim // 8
        feat_dyn_dim = 4
        self.feat_dyn_dim = feat_dyn_dim
        self.feat_cte_dim = feat_dim - feat_dyn_dim

        '''Flags'''
        self.cte_app = True
        self.compositional = False
        self.with_u = True
        self.deriv_in_state = False
        self.num_sys = num_sys
        self.composed_systems = True if self.num_sys > 0 else False

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

        self.spatial_tf = SpatialTransformation(self.cte_resolution, self.ori_resolution, out_channels=out_channels)

        self.linear_f_cte_post = nn.Linear(2 * self.feat_cte_dim, 2 * self.feat_cte_dim)
        self.linear_f_dyn_post = nn.Linear(2 * self.feat_dyn_dim, 2 * self.feat_dyn_dim)
        #
        self.bc_decoder = False                                                    #2 *
        if self.bc_decoder:
            self.image_decoder = ImageBroadcastDecoder(self.feat_cte_dim, out_channels, resolution=(16, 16)) # resolution=self.att_resolution
        else:
            self.image_decoder = ImageDecoder(self.feat_cte_dim, out_channels, dyn_dim = self.feat_dyn_dim)
        self.image_encoder = ImageEncoder(in_channels, 2 * self.feat_cte_dim, self.feat_dyn_dim, self.att_resolution, self.n_objects, ngf, n_layers, cte_app=self.cte_app, bs=batch_size)  # feat_dim * 2 if sample here
        self.koopman = KoopmanOperators(feat_dyn_dim, nf_particle, nf_effect, g_dim, u_dim, n_timesteps, deriv_in_state=self.deriv_in_state,
                                        num_sys=num_sys, alpha=1, init_scale=1)
        self.count = 1
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

        if self.deriv_in_state and self.n_timesteps > 2:
            d_x = new_x[..., 1:] - new_x[..., :-1]
            dd_x = d_x[..., 1:] - d_x[..., :-1]
            new_x = torch.cat([new_x, d_x, dd_x], dim=-1)

        return new_x.reshape(-1, new_T, new_x.shape[-2] * new_x.shape[-1]), new_T


    def forward(self, input, epoch):
        # Note: Add annealing in SPACE
        bs, T, ch, h, w = input.shape
        output = {}

        # Model reduction.
        # K = 5
        if self.count - epoch == 0:
            self.koopman.print_SV()
            # self.koopman.limit_rank_k(k=K)
            # print('rank of A is ' + str(K) + '.')
            self.count += 1

        free_pred = self.free_pred #Try again with trained.
        returned_post = []
        f_bb = input

        # Dynamic features
        T_inp = T
        f_dyn, shape, f_cte, confi = self.image_encoder(f_bb[:, :T_inp], block='dyn_track')
        f_dyn = f_dyn.reshape(-1, f_dyn.shape[-1]) # TODO: Reconstruct from here directly.

        # Sample dynamic features or reshape
        # Option 1: Don't sample dyn features
        f_dyn = f_dyn.reshape(bs * self.n_objects, T, -1)
        # Option 2: Sample dyn features
        # f_mu_dyn, f_logvar_dyn = self.linear_f_dyn_post(f_dyn).reshape(bs, self.n_objects, T_inp, -1).chunk(2, -1)
        # f_dyn_post = Normal(f_mu_dyn, F.softplus(f_logvar_dyn))
        # f_dyn = f_dyn_post.rsample()
        # f_dyn = f_dyn.reshape(bs * self.n_objects, T_inp, -1)
        # returned_post.append(f_dyn_post)

        # Get delayed dynamic features
        f_dyn_state, T_inp = self._get_full_state_hankel(f_dyn, T_inp)

        # Get inputs from delayed dynamic features

        # if not self.with_u:
        #     u, u_post, u_logit = self.koopman.to_u(f_dyn_state, temp=self.ini_alpha + epoch * self.incr_alpha, ignore=True)
        #     u = torch.zeros_like(u)
        # else:
        #     u, u_post, u_logit = self.koopman.to_u(f_dyn_state, temp=self.ini_alpha + epoch * self.incr_alpha, ignore=False)
        #     u_logit = u_logit.reshape(bs, self.n_objects, -1, u_logit.shape[-1])
        # output["u_bern_logit"] = u_logit

        # Get observations from delayed dynamic features # TODO: We could map this to double of g. The other half being a one-hot vector. No need to sparsify

        g_mask_logit = None
        if not self.compositional:
            g = self.koopman.to_g(f_dyn_state.reshape(bs * self.n_objects * T_inp, -1), self.psteps)
            g = g.reshape(bs * self.n_objects, T_inp, *g.shape[1:])
        # else:
        #     g, g_mask, g_mask_logit = self.koopman.to_composed_g(f_dyn_state.reshape(bs * self.n_objects * T_inp, -1), self.psteps)
        #     g = g.reshape(bs * self.n_objects, T_inp, *g.shape[1:])
        #     g_mask = g_mask.reshape(bs * self.n_objects, T_inp, *g_mask.shape[1:])[:, :1]
        #     g = g * g_mask
        output["g_bern_logit"] = g_mask_logit

        # Get shifted observations for sys ID
        randperm = torch.arange(g.shape[0]) # No permutation
        # randperm = torch.randperm(g.shape[0]) # No permutation

        # Sys ID
        selector = None
        A, A_inv, AA_inv, B = self.koopman.get_A_Ainv(get_A_inv=False, get_B=True)
        output["selector"] = selector
        output["Ainv"] = A_inv
        output["AAinv"] = AA_inv

        # Rollout from start_step onwards.
        output["u"] = None
        output["u_rec"] = None
        start_step = 2 # g and u must be aligned!!
        # TODO: Accumulate, take initial conditions. With NN.
        if self.with_u:
            S_for_pred, G_for_pred, u, u_logit = self.koopman.simulate(T=T_inp-start_step-1, g=g[:, start_step], u=None, B=None)
            output["u"] = u.reshape(bs * self.n_objects, -1, u.shape[-1])
        else:
            S_for_pred, G_for_pred, u, u_logit = self.koopman.simulate_no_input(T=T_inp-start_step-1, g=g[:, start_step], u=None, B=None)
        output["u_bern_logit"] = u_logit

        G_for_pred_rev = None
        # G_for_pred_rev =    self.koopman.simulate(T=T_inp-start_step-2, g=G_for_pred[:,-1], u=u[:,start_step:].flip(1), B=B, backward=True).flip(1) # Check
        # G_for_pred_rev = torch.cat([G_for_pred_rev, G_for_pred[:,-1:]], dim=1) # TODO: Check that this is ok.
        # G_for_pred_rev =    self.koopman.simulate(T=T_inp-start_step-2, g=g[:,-1], u=u[:,start_step:].flip(1), B=B, backward=True).flip(1) # Check
        # G_for_pred_rev = torch.cat([G_for_pred_rev, g[:,-1:]], dim=1) # TODO: Check that this is ok.
        # G_for_pred_rev =    self.koopman.simulate(T=T_inp-free_pred-1, g=g.flip(1)[:,free_pred],  u=u.flip(1)[:,free_pred], B=B, backward=True).flip(1) # From original pose
        # G_for_pred_rev = G_for_pred_rev.reshape(bs * self.n_objects, -1, G_for_pred_rev.shape[-1])
        output["obs_rec_pred_rev"] = G_for_pred_rev

        cases = []
        # cases.append({ "obs": None,
        #                "pose": f_dyn,
        #                "confi": confi,
        #                "shape": shape,
        #                "f_cte": f_cte,
        #                "T": T,
        #                "name": "rec_ori"})
        cases.append({"obs": g,
                      "confi": confi[:, :, self.n_timesteps-1:self.n_timesteps-1 + T_inp],
                      "shape": shape[:, :, self.n_timesteps-1:self.n_timesteps-1 + T_inp],
                      "f_cte": f_cte[:, :, self.n_timesteps-1:self.n_timesteps-1 + T_inp],
                      "T": T_inp,
                      "name": "rec"})
        cases.append({#"obs": G_for_pred,
                      "obs": None,
                      "pose": S_for_pred,
                "confi": confi[:, :, -S_for_pred.shape[1]:],
                "shape": shape[:, :, -S_for_pred.shape[1]:],
                "f_cte": f_cte[:, :, -S_for_pred.shape[1]:],
                "T": S_for_pred.shape[1],
                "name": "pred"})
        # cases.append({"obs": G_for_pred[:, -free_pred:],
        #             "confi": confi[:, :, -free_pred:],
        #             "shape": shape[:, :, -free_pred:],
        #             "f_cte": f_cte[:, :, -free_pred:],
        #             "T": free_pred,
        #             "name": "pred"})
        # cases.append({"obs": G_for_pred_rev,
        #             "confi": confi[:, :, -G_for_pred_rev.shape[1]:],
        #             "shape": shape[:, :, -G_for_pred_rev.shape[1]:],
        #             "f_cte": f_cte[:, :, -G_for_pred_rev.shape[1]:],
        #             "T": G_for_pred_rev.shape[1],
        #             "name": "pred_rev"})
        # cases.append({"obs": G_for_pred_rev,
        #             "confi": confi[:, :, :G_for_pred_rev.shape[1]],
        #             "shape": shape[:, :, :G_for_pred_rev.shape[1]],
        #             "f_cte": f_cte[:, :, :G_for_pred_rev.shape[1]],
        #             "T": G_for_pred_rev.shape[1],
        #             "name": "pred_rev"})
        outs = {}
        f_dyns = {}
        # Recover partial shape with decoded dynamical features. Iterate with new estimates of the appearance.
        # Note: This process could be iterative.
        for idx, case in enumerate(cases):

            case_name = case["name"]

            if case["obs"] is None:
                f_dyn_state = case["pose"].reshape(-1, case["pose"].shape[-1])
            else:
                f_dyn_state = self.koopman.to_s(gcodes=_get_flat(case["obs"]),
                                          psteps=self.psteps)
                u_rec = self.koopman.u_dynamics(f_dyn_state, u=None, propagate=False)[0]
                output["u_rec"] = u_rec.reshape(bs * self.n_objects, case["T"], -1)
            f_dyns[case_name] = f_dyn_state.reshape(bs * self.n_objects, case["T"], -1)

            # Encode with raw pose and confidence features
            # pose = f_dyn_state.tanh()
            # pose = torch.clamp(f_dyn_state, min=-1, max=1)

            # TODO: reflect instead of clamp - Tested
            pose  = f_dyn_state
            pose_ov_1 = F.relu(-(1 - F.relu(pose)))
            pose_un_neg1 = F.relu(-(1 - F.relu(-pose)))
            pose = pose - 2*pose_ov_1 + 2*pose_un_neg1


            # appearance features
            confi = case["confi"].reshape(bs * self.n_objects * case["T"], -1).tanh().abs()

            # Sample appearance (we call appearance features f_cte)
            if self.cte_app:
                f_cte = case["f_cte"][:, :, 0].reshape(bs * self.n_objects, -1)
                f_mu_cte, f_logvar_cte = self.linear_f_cte_post(f_cte).reshape(bs, self.n_objects, 1, -1).chunk(2, -1)
                f_cte_post = Normal(f_mu_cte, F.softplus(f_logvar_cte))
                f_cte = f_cte_post.rsample().repeat_interleave(case["T"], dim=2).reshape(bs * self.n_objects * case["T"], -1)
            else:
                f_cte = case["f_cte"].reshape(bs * self.n_objects * case["T"], -1)
                f_mu_cte, f_logvar_cte = self.linear_f_cte_post(f_cte).reshape(bs, self.n_objects, case["T"], -1).chunk(2, -1)
                f_cte_post = Normal(f_mu_cte, F.softplus(f_logvar_cte))
                # f_cte_post = Normal(f_mu_cte[:, :, 0:1], F.softplus(f_logvar_cte[:, :, 0:1])) # TODO: KL in only 1 timestep
                f_cte = f_cte_post.rsample().reshape(bs * self.n_objects * case["T"], -1)

            # Register statistics
            returned_post.append(f_cte_post)

            # Get full feature vector
            f = f_cte

            # Get output with decoder
            dec_obj = self.image_decoder(f, block='to_x')
            if not self.bc_decoder:
                grid, area = self.spatial_tf(confi, pose)
                use_confi = True
                # if case_name == "pred":
                #     use_confi = False
                outs[case_name], out_shape = self.spatial_tf.warp_and_render(dec_obj, case["shape"], confi, grid, use_confi=use_confi)
            else:
                outs[case_name] = dec_obj * confi[..., None, None]
        output["dyn_features"] = f_dyns

        ''' Returned variables '''
        returned_g = torch.cat([g, G_for_pred], dim=1) # Observations


        # TODO: this process is unnecessary, can be done in the loop
        # out_rec_ori = outs["rec_ori"]
        out_rec = outs["rec"]
        out_pred = outs["pred"]
        # out_pred_rev = outs["pred_rev"]
        #
        # output["rec_ori"] = torch.clamp(torch.sum(out_rec_ori.reshape(bs, self.n_objects, -1, *out_rec_ori.size()[1:]), dim=1), min=0, max=1)
        output["rec_ori"] = None
        output["rec"] = torch.clamp(torch.sum(out_rec.reshape(bs, self.n_objects, -1, *out_rec.size()[1:]), dim=1), min=0, max=1)
        output["pred"] = torch.clamp(torch.sum(out_pred.reshape(bs, self.n_objects, -1, *out_rec.size()[1:]), dim=1), min=0, max=1)
        output["pred_rev"] = None
        # output["pred_rev"] = torch.clamp(torch.sum(out_pred_rev.reshape(bs, self.n_objects, -1, *out_rec.size()[1:]), dim=1), min=0, max=1)

        output["obs_rec_pred"] = returned_g.reshape(bs * self.n_objects, -1, returned_g.shape[-1])
        output["gauss_post"] = returned_post
        output["A"] = A
        output["B"] = B

        # Option 1: one object mapped to 0
        # shape_o0 = o[0].shape
        # o[0] = o[0].reshape(bs, self.n_objects, *o[0].shape[1:])
        # o[0][:,0] = o[0][:,0]*0
        # o[0] = o[0].reshape(*shape_o0)

        return output