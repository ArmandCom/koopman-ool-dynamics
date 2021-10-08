import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from base import BaseModel
from utils import util as ut
from model.networks_space import ImageEncoder, ImageDecoder
from model.networks_space.relational_koopman_AB import KoopmanOperators
from model.networks_space.renderer import SpatialTransformation
from torch.distributions import Normal, kl_divergence
from utils.util import linear_annealing
from functools import reduce
from torch.autograd import Variable
from torch.distributions.categorical import Categorical
import random
import matplotlib.pyplot as plt

class RecKoopmanModel(BaseModel):
    def __init__(self, in_channels, feat_dim, nf_particle, nf_effect, g_dim, r_dim, u_dim,
                 n_objects, free_pred = 1, I_factor=10, n_blocks=1, psteps=1, n_timesteps=1, ngf=8, image_size=[64, 64], with_interactions=False, with_u=False, batch_size=40, collision_margin=None, cte_app = True):
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
        self.with_u = with_u
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
        self.n_objects = n_objects

        self.spatial_tf = SpatialTransformation(self.cte_resolution, self.image_size, out_channels=out_channels, n_objects=self.n_objects)

        self.linear_f_cte_post = nn.Linear(2 * self.feat_cte_dim, 2 * self.feat_cte_dim)
        # self.linear_f_dyn_post = nn.Linear(self.feat_dyn_dim, 4)

        self.image_decoder = ImageDecoder(self.feat_cte_dim, out_channels + 1, dyn_dim = self.feat_dyn_dim)

        self.image_encoder = ImageEncoder(in_channels, 2 * self.feat_cte_dim, self.feat_dyn_dim, self.att_resolution, self.n_objects, ngf, image_size, n_timesteps=n_timesteps, cte_app=self.cte_app, bs=batch_size)  # feat_dim * 2 if sample here
        self.koopman = KoopmanOperators(feat_dyn_dim, nf_particle, nf_effect, g_dim, r_dim, u_dim, n_timesteps, deriv_in_state=self.deriv_in_state,
                                        with_interactions = with_interactions, init_scale=1, collision_margin=collision_margin)

        self.count = 1

    def forward(self, input, epoch_iter, test=False):
        # Note: Add annealing in SPACE
        bs, T, ch, h, w = input.shape

        # T = linear_annealing(input.device, epoch_iter[1], start_step=2000, end_step=25000, start_value=8, end_value=T)
        # T = int(T)
        # input = input[:, :T]

        output = {}
        if epoch_iter[0] is not -1:
            temp = linear_annealing(input.device, epoch_iter[1], start_step=4000, end_step=15000, start_value=1, end_value=0.1)
            inputs_prob = linear_annealing(input.device, epoch_iter[1], start_step=2000, end_step=4000, start_value=0.0, end_value=0.0)

            if self.count < epoch_iter[0]:
                self.count = epoch_iter[0]
            if self.count == epoch_iter[0]:
                self.koopman.print_SV()
                self.count += 1
        else:
            temp = 0.1
            inputs_prob = 0.0
        # A_std = linear_annealing(input.device, epoch_iter[1], start_step=0, end_step=10000, start_value=0.5, end_value=0)
        # if A_std != 0:
        #     self.koopman.add_noise_to_A(A_std)

        # if self.collision_margin is not None:
        #     print('You are using collision margin idiot')
        #     exit()
        #     self.koopman.collision_margin = linear_annealing(input.device, epoch_iter[1], start_step=1000, end_step=8000, start_value=0.5, end_value=0.22)

        # Model reduction.
        # K = 5

        returned_post = []
        f_bb = input

        # Dynamic features
        T_inp = T
        f_dyn, shape, f_cte, confi, layers = self.image_encoder(f_bb[:, :T_inp], block='dyn_track')
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
        f_dyn_state, T_inp = self.koopman.get_full_state_hankel(f_dyn, T_inp) # T-2

        # f_dyn_state = f_dyn_state.reshape(bs, self.n_objects, T_inp, -1).permute(0, 2, 1, 3).reshape(bs * T_inp, self.n_objects, -1)
        # ut.print_var_shape(f_dyn_state, 'f_dyn_state')

        # Start with randomsearch of the action space
        if inputs_prob == 0.0:
            inputs = None
        else:
            if random.random() < inputs_prob:
                inputs = torch.randint(0, 2, [bs * T_inp, self.n_objects, 1]).float().to(f_dyn_state.device)
            else:
                inputs = None

        g_stack, u_stack, u_tp_rec, sel_tp_rec, selectors = self.koopman.to_g(f_dyn_state.reshape(bs, self.n_objects, T_inp, -1), temp=temp, inputs=inputs, with_u=self.with_u)
        g = g_stack.transpose(-2, -1).flatten(start_dim=-2)
        # inps = u_stack.transpose(-2, -1).flatten(start_dim=-2)
        # ut.print_var_shape(g, 'g_out')


        g_mask_logit = None
        output["g_bern_logit"] = g_mask_logit

        # Sys ID
        # A = self.koopman.get_A()
        A,B = self.koopman.get_AB()

        # Rollout from start_step onwards.
        output["u"] = None
        output["sel"] = None
        output["u_rec"] = None
        output["sel_rec"] = None

        # 1 by 1 steps
        g_pred_1step = self.koopman.rollout_1step(g_stack, u_stack)
        output["g_pred_1step"] = g_pred_1step.reshape(bs*self.n_objects, -1, g.shape[-1])
        state_tp = self.koopman.inv_mapping(g_pred_1step)
        output["s_pred_1step"], confi_1step = state_tp[0].reshape(bs*self.n_objects, -1, f_dyn_state.shape[-1]), state_tp[1].reshape(bs, self.n_objects, -1, 1)

        _, u_pred_1_step_rec, _, _, _ = self.koopman.to_g(
            output["s_pred_1step"].reshape(bs, self.n_objects, -1, f_dyn_state.shape[-1]), temp=temp, inputs=inputs, with_u=self.with_u)
        g_pred_2step = self.koopman.rollout_1step(g_pred_1step, u_pred_1_step_rec)
        output["g_pred_2step"] = g_pred_2step.reshape(bs*self.n_objects, -1, g.shape[-1])
        state_tp = self.koopman.inv_mapping(g_pred_2step)
        output["s_pred_2step"], confi_2step = state_tp[0].reshape(bs*self.n_objects, -1, f_dyn_state.shape[-1]), state_tp[1].reshape(bs, self.n_objects, -1, 1)

        start_step = 0 # g and u must be aligned!!
        init_g = g_stack[:, :, start_step:start_step+1]

        T_sim = T_inp-start_step-1 # T-2-1-*
        S_for_pred, confi_for_pred, G_for_pred, u_tp, sel_tp, selectors_for_pred = self.koopman.simulate(T=T_sim, g=init_g, inputs=None, temp=temp, logit_out=False, with_u=self.with_u) #[B, N, T - n_ts - start_step, -1]
        # ut.print_var_shape([S_for_pred, G_for_pred, u_tp[0]], 's, g, us after rollout', q=True)

        if self.with_u and u_tp_rec[0] is not None:
            output["u"] = u_tp_rec[0].reshape(bs * self.n_objects, -1, u_tp_rec[0].shape[-1])[:, :, -S_for_pred.shape[2]:] #TODO: Check! Used to be u.
            output["u_rec"] = u_tp_rec[0].reshape(bs * self.n_objects, -1, u_tp_rec[0].shape[-1])
        if self.with_interactions and sel_tp_rec[0] is not None and sel_tp[0] is not None:
            output["sel"] = sel_tp[0].reshape(bs * self.n_objects, -1, self.n_objects)
            output["sel_rec"] = sel_tp_rec[0].reshape(bs * self.n_objects, -1, self.n_objects)

        output["sel_bern_logit"] = sel_tp[1]
        output["u_bern_logit"] = u_tp[1]

        G_for_pred_rev = None
        output["obs_rec_pred_rev"] = G_for_pred_rev

        cases = []

        # cases.append({"obs": None,
        #               "pose": f_dyn_state.reshape(bs, self.n_objects, *f_dyn_state.shape[1:]),
        #               "confi": confi[:, :, -f_dyn_state.shape[1]:],
        #               "shape": shape[:, :, -f_dyn_state.shape[1]:],
        #               "f_cte": f_cte[:, :, -f_dyn_state.shape[1]:],
        #               "T": f_dyn_state.shape[1],
        #               "name": "rec_ori"})
        # cases.append({"obs": g_stack,
        #               "confi": confi[:, :, self.n_timesteps-1:self.n_timesteps-1 + T_inp],
        #               "shape": shape[:, :, self.n_timesteps-1:self.n_timesteps-1 + T_inp],
        #               "f_cte": f_cte[:, :, self.n_timesteps-1:self.n_timesteps-1 + T_inp],
        #               "T": T_inp,
        #               "name": "rec"})

        cases.append({"obs": None,
                      "input": input,
                      "f_dyn_state": f_dyn_state.reshape(bs, self.n_objects, *f_dyn_state.shape[1:]),
                      "pose": f_dyn.reshape(bs, self.n_objects, *f_dyn.shape[1:]),
                      "confi": confi,
                      "shape": shape,
                      "layers": layers,
                      "f_cte": f_cte,
                      "T": f_dyn.shape[1],
                      "name": "rec_ori"})
        # T_pred = output["s_pred_1step"].shape[1] - 1
        # cases.append({#"obs": G_for_pred,
        #     "obs": None,
        #     "pose": output["s_pred_1step"].reshape(bs, self.n_objects, T_pred + 1, -1)[..., :T_pred, :],
        #     "confi": confi_1step[:, :, :T_pred], #confi[:, :, -S_for_pred.shape[2]:],
        #     "shape": shape[:, :, -T_pred:],
        #     "f_cte": f_cte[:, :, -T_pred:],
        #     "T": T_pred,
        #     "name": "pred"})

        # Note: These two are activated for experiments
        # cases.append({
        #     "obs": None,
        #     "pose": S_for_pred,
        #     "confi": confi[:, :, -S_for_pred.shape[2]:],#confi_for_pred, #confi[:, :, -S_for_pred.shape[2]:],
        #     "shape": shape[:, :, -S_for_pred.shape[2]:],
        #     "f_cte": f_cte[:, :, -S_for_pred.shape[2]:],
        #     "T": S_for_pred.shape[2],
        #     "name": "pred_roll"})

        # T_pred = output["s_pred_2step"].shape[1] - 2
        # cases.append({#"obs": G_for_pred,
        #     "obs": None,
        #     "pose": output["s_pred_2step"].reshape(bs, self.n_objects, T_pred + 2, -1)[..., :T_pred, :],
        #     "confi": confi_2step[:, :, :T_pred], #confi[:, :, -S_for_pred.shape[2]:],
        #     "shape": shape[:, :, -T_pred:],
        #     "f_cte": f_cte[:, :, -T_pred:],
        #     "T": T_pred,
        #     "name": "pred"})

        outs = {}
        f_dyns = {}
        f_dyns["rec_ori"] = f_dyn_state

        # Recover partial shape with decoded dynamical features. Iterate with new estimates of the appearance.
        # Note: This process could be iterative.
        for idx, case in enumerate(cases):

            case_name = case["name"]
            output['pose'] = case["pose"]

            if case["obs"] is None:
                f_dyn_out = case["pose"].reshape(-1, case["pose"].shape[-1])
                # ut.print_var_shape([case["pose"], f_dyn_out], 'State')
            else:
                f_dyn_out, case["confi"] = self.koopman.to_s(gcodes=case["obs"],
                                          psteps=self.psteps)
                f_dyn_out = f_dyn_out.flatten(start_dim=0, end_dim=2)
                # ut.print_var_shape([case["obs"], f_dyn_out], 'Obs and state for rec')

            f_dyns[case_name] = f_dyn_out.reshape(bs * self.n_objects, case["T"], -1)

            # Note: linear
            if case_name is 'rec_ori':
                pose = f_dyn_out.reshape(bs * self.n_objects * case["T"], -1).tanh()
                f_dyns[case_name] = case['f_dyn_state'].reshape(bs * self.n_objects, case["T"] -self.n_timesteps +1, -1)
            else:
                pose = f_dyn_out.reshape(bs * self.n_objects * case["T"], -1, self.n_timesteps)[..., 0].tanh()


            rand_pose = torch.randn(bs, self.n_objects, 1, 2)\
                .repeat(1, 1, case["T"], 1).reshape(bs * self.n_objects * case["T"], -1).to(pose.device)
            rand_pose = torch.cat([rand_pose, torch.zeros_like(rand_pose)], dim=-1)
            pose = pose + rand_pose
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

            # f_mu_dyn, f_logvar_dyn = self.linear_f_dyn_post(pose).reshape(bs, self.n_objects, case["T"], -1).chunk(2, -1)
            # pose_post = Normal(f_mu_dyn, F.softplus(f_logvar_dyn))
            # pose = pose_post.rsample().reshape(bs * self.n_objects * case["T"], -1).tanh()

            # Register statistics
            # returned_post.append(pose_post)
            returned_post.append(f_cte_post)

            # Get full feature vector
            f = f_cte

            # Get output with decoder
            dec_obj, dec_shape = self.image_decoder(f, block='to_x')
            #Note: before and (case_name == 'pred' or case_name == 'pred_roll')
            if test : #TODO: Change to or in case of
                # Use last appearance
                # dec_obj_T = dec_obj.reshape(bs * self.n_objects, case["T"], *dec_obj.shape[1:])[:,0:1]
                # dec_obj = dec_obj_T.repeat_interleave(case["T"], dim=1).reshape(bs * self.n_objects * case["T"], *dec_obj.shape[1:])
                use_confi = False
            else:
                use_confi = True # TODO: Originally true
            grid, area = self.spatial_tf.get_sampling_grid(confi, pose)
            area = area.view(bs, case["T"], self.n_objects, 1).mean(2) # N * T * 1
            outs[case_name] = \
                self.spatial_tf\
                    (confi, case["layers"], dec_shape, dec_obj, grid, Y_b = case["input"])
            output['shape'] = dec_shape

        output["dyn_features"] = f_dyns

        ''' Returned variables '''
        returned_g = torch.cat([g, G_for_pred.transpose(-2, -1).flatten(start_dim=-2)], dim=2) # Observations

        # Option 1: Old way
        # out_rec_ori = outs["rec_ori"]
        # # out_rec = outs["rec"]
        # out_pred = outs["pred"]
        # out_pred_roll = outs["pred_roll"]
        # # out_pred_rev = outs["pred_rev"]
        #
        # # output["rec_ori"] = torch.clamp(torch.sum(out_rec_ori.reshape(bs, self.n_objects, -1, *out_rec_ori.size()[1:]), dim=1), min=0, max=1)
        # output["rec_ori"] = torch.clamp(torch.sum(out_rec_ori.reshape(bs, self.n_objects, -1, *out_rec_ori.size()[1:]), dim=1), min=0, max=1)
        # output["rec"] = None
        # # output["rec"] = torch.clamp(torch.sum(out_rec.reshape(bs, self.n_objects, -1, *out_rec.size()[1:]), dim=1), min=0, max=1)
        # output["pred"] = torch.clamp(torch.sum(out_pred.reshape(bs, self.n_objects, -1, *out_pred.size()[1:]), dim=1), min=0, max=1)
        # output["pred_roll"] = torch.clamp(torch.sum(out_pred_roll.reshape(bs, self.n_objects, -1, *out_pred.size()[1:]), dim=1), min=0, max=1)
        # # output["pred_roll"] = None

        # TODO: For key in outs do this (or hardcoded keys and else = None)
        # output["rec_ori"] = torch.clamp(torch.sum(outs["rec_ori"].reshape(bs, self.n_objects, -1, *outs["rec_ori"].size()[1:]), dim=1), min=0, max=1)
        output["rec_ori"] = outs["rec_ori"]
        output["pred"] = None
        output["pred_roll"] = None
        output["rec"] = None

        # Note: WIth bouncing balls just sum them up?

        output["obs_rec_pred"] = returned_g.reshape(bs * self.n_objects, -1, returned_g.shape[-1])
        output["gauss_post"] = returned_post
        output["A"] = A
        output["B"] = B
        output["selectors_rec"] = selectors.reshape(bs * self.n_objects, -1, selectors.shape[-1])
        output["selectors_pred"] = selectors_for_pred.reshape(bs * self.n_objects, -1, selectors_for_pred.shape[-1]) if selectors_for_pred is not None else None

        # Option: one object mapped to 0 for ablation and test
        # shape_o0 = o[0].shape
        # o[0] = o[0].reshape(bs, self.n_objects, *o[0].shape[1:])
        # o[0][:,0] = o[0][:,0]*0
        # o[0] = o[0].reshape(*shape_o0)

        return output