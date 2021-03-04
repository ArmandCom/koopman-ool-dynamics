import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from base import BaseModel
from utils import util as ut
from model.networks_space import ImageEncoder, ImageDecoder, ImageBroadcastDecoder
from model.networks_space.relational_koopman_select import KoopmanOperators
from model.networks_space.spatial_tf import SpatialTransformation
from torch.distributions import Normal, kl_divergence
from utils.util import linear_annealing
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
    def __init__(self, in_channels, feat_dim, nf_particle, nf_effect, g_dim, r_dim, u_dim,
                 n_objects, free_pred = 1, I_factor=10, n_blocks=1, psteps=1, n_timesteps=1, ngf=8, image_size=[64, 64], with_interactions=False, batch_size=40, collision_margin=None, cte_app = True):
        super().__init__()
        out_channels = in_channels
        n_layers = int(np.log2(image_size[0])) - 1

        self.u_dim = u_dim
        self.r_dim = r_dim

        # Set state dim with config, depending on how many time-steps we want to take into account
        self.image_size = image_size
        self.n_timesteps = n_timesteps
        self.state_dim = feat_dim
        self.I_factor = I_factor
        self.psteps = psteps
        self.g_dim = g_dim
        self.free_pred = free_pred
        self.collision_margin = collision_margin

        # feat_dyn_dim = feat_dim // 8
        feat_dyn_dim = 4 # TODO: This can be higher and then only use 4 of the dimensions for reconstruction. The rest might be related to physics constants.
        self.feat_dyn_dim = feat_dyn_dim
        self.feat_cte_dim = feat_dim - feat_dyn_dim

        '''Flags'''
        self.cte_app = cte_app
        self.with_u = True
        self.deriv_in_state = False
        self.with_interactions = with_interactions

        self.ini_alpha = 1
        # Note:
        #  - I leave it to 0 now. If it increases too fast, the gradients might be affected
        self.incr_alpha = 0.1


        self.cte_resolution = (32, 32)
        self.ori_resolution = image_size
        self.att_resolution = (16, 16)
        self.obj_resolution = (4, 4)
        # self.n_objects = reduce((lambda x, y: x * y), self.obj_resolution)
        self.n_objects = n_objects

        self.spatial_tf = SpatialTransformation(self.cte_resolution, self.image_size, out_channels=out_channels)

        self.linear_f_cte_post = nn.Linear(2 * self.feat_cte_dim, 2 * self.feat_cte_dim)
        self.linear_f_dyn_post = nn.Linear(2 * self.feat_dyn_dim, 2 * self.feat_dyn_dim)
        #
        self.bc_decoder = False                                                    #2 *
        if self.bc_decoder:
            self.image_decoder = ImageBroadcastDecoder(self.feat_cte_dim, out_channels, resolution=(16, 16)) # resolution=self.att_resolution
        else:
            self.image_decoder = ImageDecoder(self.feat_cte_dim, out_channels, dyn_dim = self.feat_dyn_dim)
        self.image_encoder = ImageEncoder(in_channels, 2 * self.feat_cte_dim, self.feat_dyn_dim, self.att_resolution, self.n_objects, ngf, image_size, cte_app=self.cte_app, bs=batch_size)  # feat_dim * 2 if sample here
        self.koopman = KoopmanOperators(feat_dyn_dim, nf_particle, nf_effect, g_dim, r_dim, u_dim, n_timesteps, deriv_in_state=self.deriv_in_state,
                                        with_interactions = with_interactions, init_scale=1, collision_margin=collision_margin)

        self.count = 1

    def forward(self, input, epoch_iter):
        # Note: Add annealing in SPACE
        bs, T, ch, h, w = input.shape
        output = {}
        temp = linear_annealing(input.device, epoch_iter[1], start_step=4000, end_step=15000, start_value=1, end_value=0.001)
        if self.collision_margin is not None:
            self.koopman.collision_margin = linear_annealing(input.device, epoch_iter[1], start_step=1000, end_step=8000, start_value=0.5, end_value=0.22)
        # Model reduction.
        # K = 5
        if self.count - epoch_iter[0] == 0:
            self.koopman.print_SV()
            # self.koopman.limit_rank_k(k=K)
            # self.koopman.limit_rank_th(th=0.1)
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
        f_dyn_state, T_inp = self.koopman.get_full_state_hankel(f_dyn, T_inp)

        # f_dyn_state = f_dyn_state.reshape(bs, self.n_objects, T_inp, -1).permute(0, 2, 1, 3).reshape(bs * T_inp, self.n_objects, -1)
        # ut.print_var_shape(f_dyn_state, 'f_dyn_state')

        g_stack, u_tp_rec, sel_tp_rec, selectors = self.koopman.to_g(f_dyn_state.reshape(bs, self.n_objects, T_inp, -1), temp=temp)
        g = g_stack.sum(-1)
        # ut.print_var_shape(g, 'g_out')


        g_mask_logit = None
        output["g_bern_logit"] = g_mask_logit

        # Sys ID
        A = self.koopman.get_A()

        # Rollout from start_step onwards.
        output["u"] = None
        output["sel"] = None
        output["u_rec"] = None
        output["sel_rec"] = None

        # TODO: Inverse dynamics?
        g_pred_1step = self.koopman.rollout_1step(g_stack)
        output["g_pred_1step"] = g_pred_1step.reshape(bs*self.n_objects, -1, g.shape[-1])
        output["s_pred_1step"] = self.koopman.inv_mapping(g_pred_1step).reshape(bs*self.n_objects, -1, f_dyn.shape[-1])

        start_step = 6 # g and u must be aligned!!
        init_s = f_dyn.reshape(bs, self.n_objects, -1, f_dyn.shape[-1])[:, :, start_step:start_step+self.n_timesteps-1] # Last time step will be the converted g.
        init_g = g[:, :, start_step:start_step+1]

        logit_out = True if self.collision_margin is None else False
        T_sim = T_inp-start_step-1
        # u_sim = u_tp_rec[0][..., -T_sim:, :]
        u_sim = None
        # TODO: Return full state instead of current
        S_for_pred, G_for_pred, u_tp, sel_tp, selectors_for_pred = self.koopman.simulate(T=T_sim, g=init_g, init_s=init_s, inputs=u_sim, temp=temp, logit_out=logit_out) #[B, N, T - n_ts - start_step, -1]
        # ut.print_var_shape([S_for_pred, G_for_pred, u_tp[0]], 's, g, us after rollout', q=True)

        if self.with_u and u_tp_rec[0] is not None:
            output["u"] = u_tp_rec[0].reshape(bs * self.n_objects, -1, u_tp_rec[0].shape[-1])[:, :, -S_for_pred.shape[2]:] #TODO: Check! Used to be u.
            output["u_rec"] = u_tp_rec[0].reshape(bs * self.n_objects, -1, u_tp_rec[0].shape[-1])
        if self.with_interactions:
            output["sel"] = sel_tp[0].reshape(bs * self.n_objects, -1, self.n_objects)
            output["sel_rec"] = sel_tp_rec[0].reshape(bs * self.n_objects, -1, self.n_objects)

        output["sel_bern_logit"] = sel_tp[1]
        output["u_bern_logit"] = u_tp[1]

        G_for_pred_rev = None
        output["obs_rec_pred_rev"] = G_for_pred_rev

        cases = []
        # cases.append({"obs": None,
        #               "pose": f_dyn.reshape(bs, self.n_objects, *f_dyn.shape[1:]),
        #               "confi": confi,
        #               "shape": shape,
        #               "f_cte": f_cte,
        #               "T": f_dyn.shape[1],
        #               "name": "rec_ori"})
        cases.append({"obs": g,
                      "confi": confi[:, :, self.n_timesteps-1:self.n_timesteps-1 + T_inp],
                      "shape": shape[:, :, self.n_timesteps-1:self.n_timesteps-1 + T_inp],
                      "f_cte": f_cte[:, :, self.n_timesteps-1:self.n_timesteps-1 + T_inp],
                      "T": T_inp,
                      "name": "rec"})
        cases.append({#"obs": G_for_pred,
                      "obs": None,
                      "pose": S_for_pred,
                "confi": confi[:, :, -S_for_pred.shape[2]:],
                "shape": shape[:, :, -S_for_pred.shape[2]:],
                "f_cte": f_cte[:, :, -S_for_pred.shape[2]:],
                "T": S_for_pred.shape[2],
                "name": "pred"})
        # cases.append({"obs": G_for_pred[:, -free_pred:],
        #             "confi": confi[:, :, -free_pred:],
        #             "shape": shape[:, :, -free_pred:],
        #             "f_cte": f_cte[:, :, -free_pred:],
        #             "T": free_pred,
        #             "name": "pred"})
        outs = {}
        f_dyns = {}
        f_dyns["rec_ori"] = f_dyn

        # Recover partial shape with decoded dynamical features. Iterate with new estimates of the appearance.
        # Note: This process could be iterative.
        for idx, case in enumerate(cases):

            case_name = case["name"]

            if case["obs"] is None:
                f_dyn_out = case["pose"].reshape(-1, case["pose"].shape[-1])
                # ut.print_var_shape([case["pose"], f_dyn_out], 'State')
                # print(case["T"])

            else:
                f_dyn_out = self.koopman.to_s(gcodes=case["obs"],
                                          psteps=self.psteps).flatten(start_dim=0, end_dim=2)

                # ut.print_var_shape([case["obs"], f_dyn_out], 'Obs and state for rec')
                # print(case["T"])
            f_dyns[case_name] = f_dyn_out.reshape(bs * self.n_objects, case["T"], -1)

            # Encode with raw pose and confidence features
            # Note: Clamp -1,1
            # pose = torch.clamp(f_dyn_out, min=-1, max=1)
            # Note: reflect instead of clamp
            # pose  = f_dyn_out
            # pose_ov_1 = F.relu(-(1 - F.relu(pose)))
            # pose_un_neg1 = F.relu(-(1 - F.relu(-pose)))
            # pose = pose - 2*pose_ov_1 + 2*pose_un_neg1
            # Note: Tanh
            # pose = f_dyn_out.tanh()
            # Note: After clamping
            pose = f_dyn_out

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
        returned_g = torch.cat([g, G_for_pred], dim=2) # Observations


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
        output["B"] = None
        output["selectors_rec"] = selectors.reshape(bs * self.n_objects, -1, selectors.shape[-1])
        output["selectors_pred"] = selectors_for_pred.reshape(bs * self.n_objects, -1, selectors_for_pred.shape[-1])

        # Option 1: one object mapped to 0
        # shape_o0 = o[0].shape
        # o[0] = o[0].reshape(bs, self.n_objects, *o[0].shape[1:])
        # o[0][:,0] = o[0][:,0]*0
        # o[0] = o[0].reshape(*shape_o0)

        return output