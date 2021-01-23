import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from base import BaseModel
from model.networks.kornia import get_affine_params, warp_affine, get_affine_params_from_pose, test_final_functions_affine #, test_kornia, test_kornia_affine, test_final_functions_affine
# from model.networks import ImageEncoder, SBImageDecoder, SimpleSBImageDecoder, \
#     KoopmanOperators, AttImageEncoder, ImageDecoder, \
#     deconvSpatialDecoder, linearSpatialDecoder, LinearEncoder, ObjectAttention
from model.networks import KoopmanOperators
# from model.networks_slot_attention import ImageEncoder, ImageDecoder
# from model.networks_self_attention import ImageEncoder, ImageDecoder
# from model.networks_dyn_attention import ImageEncoder, ImageDecoder
from model.networks_space import ImageEncoder, ImageDecoder, ImageBroadcastDecoder
from model.networks_space.coord_networks import KeyPointsExtractor, PoseNet
from model.networks_space.spatial_tf import SpatialTransformation

from torch.distributions import Normal, kl_divergence
from functools import reduce


# from model.networks_cswm.modules import TransitionGNN, EncoderCNNLarge, EncoderCNNMedium, EncoderCNNSmall, EncoderMLP, DecoderCNNLarge, DecoderCNNMedium, DecoderCNNSmall, DecoderMLP
from torch.autograd import Variable
from torch.distributions.categorical import Categorical
import random
import matplotlib.pyplot as plt

''' A main objective of the model is to separate the dimensions to be modeled/propagated by koopman operator
    from the constant dimensions - elements that remain constant - which should be the majority of info. 
    This way we only need to control a reduced amount of dimensions - and we can make them variable.
    The constant factors can also work as a padding for the dimensionality to remain constant. '''

def _get_flat(x, keep_dim=False):
    if keep_dim:
        return x.reshape(torch.Size([1, x.size(0) * x.size(1)]) + x.size()[2:])
    return x.reshape(torch.Size([x.size(0) * x.size(1)]) + x.size()[2:])

# If we want it to be variational:
def _sample_latent_simple(mu, logvar, n_samples=1):
    std = torch.exp(0.5 * logvar)
    # std = torch.stack([std]*n_samples, dim=-1)
    eps = torch.randn_like(std)
    sample = mu + eps * std
    return sample

# def _clamp_diagonal(A, min, max):
#     # Note: this doesn't help. We need spectral normalization.
#     eye = torch.zeros_like(A)
#     ids = torch.arange(0, A.shape[-1])
#     eye[..., ids, ids] = 1
#     return A*(1-eye) + torch.clamp(A * eye, min, max)

# TODO:
#  0: Calculate As and Bs in both directions (given the observations). Forward * Backward should be the identity.
#  0: Curriculum Learning by increasing T progressively.
#  1: Invert or Sample randomly u and contrastive loss.
#  2: Treat n_timesteps with conv_nn? Or does it make sense to mix f1(t) and f2(t-1)?
#  3: Eigenvalues and Eigenvector. Fix one get the other.
#  4: Observation selector using Gumbel (should apply mask emissor and receptor
#       and it should be the same across time). Compositional koopman
#  5: Block diagonal l1(max(off_diag)) loss. Try without specifying objects.

class RecKoopmanModel(BaseModel):
    def __init__(self, in_channels, feat_dim, nf_particle, nf_effect, g_dim, u_dim,
                 n_objects, I_factor=10, n_blocks=1, psteps=1, n_timesteps=1, ngf=8, image_size=[64, 64]):
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
        # self.n_objects = n_objects

        # TODO: give as params. I'm lazy to do this now
        self.pose_limits, self.pose_bias = [4, 4, 1.5, 1.5], [0.5, 0.5, 0, 0]

        # self.softmax = nn.Softmax(dim=-1)
        # self.sigmoid = nn.Sigmoid()

        # self.initial_conditions = nn.Sequential(nn.Linear(feat_dim * n_timesteps * 2, feat_dim * n_timesteps),
        #                                         nn.ReLU(),
        #                                         nn.Linear(feat_dim * n_timesteps, g_dim * 2))

        self.content = None

        feat_dyn_dim = feat_dim // 8
        feat_dyn_dim = 4
        self.feat_dyn_dim = feat_dyn_dim
        self.feat_cte_dim = feat_dim - feat_dyn_dim

        self.with_u = False
        self.n_iters = 1
        self.ini_alpha = 1
        # Note:
        #  - I leave it to 0 now. If it increases too fast, the gradients might be affected
        self.incr_alpha = 0.1

        # Note:
        #  - May be substituted by 0s.
        #  - This noise has lower variance than prior. Otherwise it would sample regular features.
        self.f_cte_ini_std = 0.0

        self.cte_resolution = (32, 32)
        self.ori_resolution = (128, 128)
        self.att_resolution = (16, 16)
        self.obj_resolution = (4, 4)
        self.n_objects = reduce((lambda x, y: x * y), self.obj_resolution)
        self.n_objects = n_objects

        # self.rnn_f_cte = nn.LSTM(feat_dim - feat_dyn_dim, feat_dim - feat_dyn_dim, 1, bias=False, batch_first=True)
        # self.rnn_f_cte = nn.GRU(feat_dim - feat_dyn_dim, feat_dim - feat_dyn_dim, 1, bias=False, batch_first=True)

        # self.linear_f_cte_mu = nn.Linear(self.feat_cte_dim, self.feat_cte_dim)
        # self.linear_f_cte_logvar = nn.Linear(self.feat_cte_dim, self.feat_cte_dim)

        self.spatial_tf = SpatialTransformation(self.cte_resolution, self.ori_resolution)

        self.linear_f_cte_post = nn.Linear(2 * self.feat_cte_dim, 2 * self.feat_cte_dim)
        self.linear_f_dyn_post = nn.Linear(2 * self.feat_dyn_dim, 2 * self.feat_dyn_dim)
        #
        self.bc_decoder = True                                                    #2 *
        if self.bc_decoder:
            self.image_decoder = ImageBroadcastDecoder(self.feat_cte_dim, out_channels, resolution=(8,8)) # resolution=self.att_resolution
        else:
            self.image_decoder = ImageDecoder(self.feat_cte_dim, out_channels, dyn_dim = self.feat_dyn_dim, resolution=self.cte_resolution)
        self.image_encoder = ImageEncoder(in_channels, 2 * self.feat_cte_dim, self.feat_dyn_dim, self.att_resolution, self.n_objects, ngf, n_layers)  # feat_dim * 2 if sample here
        self.koopman = KoopmanOperators(feat_dyn_dim, nf_particle, nf_effect, g_dim, u_dim, n_timesteps, n_blocks)

        # self.bb_points = KeyPointsExtractor(self.feat_dyn_dim, hidden_dim=self.feat_dyn_dim * 2, resolution=self.cte_resolution, n_points=2)
        self.pose_encoder = PoseNet(self.feat_dyn_dim, pose_dim=4)

    def _get_full_state(self, x, T):

        if self.n_timesteps < 2:
            return x, T
        new_T = T - self.n_timesteps + 1
        x = x.reshape(-1, T, *x.shape[1:])
        new_x = []
        for t in range(new_T):
            new_x.append(torch.cat([x[:, t + idx]
                         for idx in range(self.n_timesteps)], dim=-1))
        # torch.cat([ torch.zeros_like( , x[:,0,0:1]) + self.t_grid[idx]], dim=-1)
        new_x = torch.stack(new_x, dim=1)
        return new_x.reshape(-1, new_x.shape[-1]), new_T

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
        # TODO: Goals: 1. multiobject using grid system, dynamic interactions using koopman.
        #  Add annealing in SPACE
        bs, T, ch, h, w = input.shape
        # Percentage of output

        # test_final_functions_affine(input)
        free_pred = T//4
        returned_post = []

        input = input.cuda()
        # Backbone deterministic features
        f_bb = self.image_encoder(input, block='backbone')

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
        # Note:
        #  - U might depend also in features from the scene. Residual features (slot n+1 / Background)
        #  - Temperature increase might not be necessary
        #  - Gumbel softmax might help
        u, u_dist = self.koopman.to_u(f_dyn_state, temp=self.ini_alpha + epoch * self.incr_alpha, ignore=True)
        if not self.with_u:
            u = torch.zeros_like(u)
        # Get observations from delayed dynamic features
        g = self.koopman.to_g(f_dyn_state.reshape(bs * self.n_objects * T_inp, -1), self.psteps)
        g = g.reshape(bs * self.n_objects, T_inp, *g.shape[1:])

        # Get shifted observations for sys ID
        randperm = torch.arange(g.shape[0]) # No permutation
        # randperm = torch.randperm(g.shape[0]) # Random permutation
        if free_pred > 0:
            G_tilde = g[randperm, :-1-free_pred, None]
            H_tilde = g[randperm, 1:-free_pred, None]
        else:
            G_tilde = g[randperm, :-1, None]
            H_tilde = g[randperm, 1:, None]

        # Sys ID
        A, B, A_inv, fit_err = self.koopman.system_identify(G=G_tilde, H=H_tilde, U=u[randperm, :T_inp-free_pred-1], I_factor=self.I_factor) # Try not permuting U when inp is permutted

        # Rollout from start_step onwards.
        start_step = 2 # g and u must be aligned!!
        #TODO: Try one more step for any of the inputs and check if it blows up
        G_for_pred = self.koopman.simulate(T=T_inp-start_step-1, g=g[:,start_step], u=u[:,start_step:], A=A, B=B)
        g_for_koop= G_for_pred

        assert f_bb[:, self.n_timesteps-1:self.n_timesteps-1 + T_inp].shape[1] == f_bb[:, self.n_timesteps-1:].shape[1]
        rec = {"obs": g,
               "confi": confi[:, :, self.n_timesteps-1:self.n_timesteps-1 + T_inp],
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
        BBs = {}

        # TODO: Check if the indices for f_bb and supervision are correct.
        # Recover partial shape with decoded dynamical features. Iterate with new estimates of the appearance.
        # Note: This process could be iterative.
        for idx, case in enumerate([rec, pred]):

            case_name = case["name"]
            # get back dynamic features
            # if case_name == "rec":
            #     f_dyn_state = f_dyn[:, -T_inp:].reshape(-1, *f_dyn.shape[2:])
            # else:
            # TODO: Try with Initial f_dyn (no koop)
            f_dyn_state = self.koopman.to_s(gcodes=_get_flat(case["obs"]),
                                          psteps=self.psteps)
            # TODO: From f_dyn extract where, scale, etc. with only Y = WX + b. Review.

            # Initial noisy (low variance) constant vector.
            # Note:
            #  - Different realization for each time-step.
            #  - Is it a Variable?
            # f_cte_ini = torch.randn(bs * self.n_objects * case["T"], self.feat_cte_dim).to(f_dyn_state.device) #* self.f_cte_ini_std

            # Get full feature vector
            # f = torch.cat([ f_dyn_state,
            #                 f_cte_ini], dim=-1)

            # Get coarse features from which obtain queries and/or decode
            # f_coarse = self.image_decoder(f, block = 'coarse')
            # f_coarse = f_coarse.reshape(bs, self.n_objects, case["T"], *f_coarse.shape[1:])

            for _ in range(self.n_iters):

                # Get constant feature vector through attention
                # f_cte = self.image_encoder(case["backbone_features"], f_coarse, block='cte')

                # Encode with raw dynamic features
                # pose = self.pose_encoder(f_dyn_state)
                pose = f_dyn_state
                # M_center, M_relocate = get_affine_params_from_pose(pose, self.pose_limits, self.pose_bias)

                # Option 2: Previous impl
                # bb_feat = case["backbone_features"].unsqueeze(1).repeat_interleave(self.n_objects, dim=1)\
                #     .reshape(-1, *case["backbone_features"].shape[-4:]) # Repeat for number of objects
                # warped_bb_feat = (bb_feat, M_center, *self.cte_resolution) # Check all resolutions are coordinate [32, 32]
                # f_cte = self.image_encoder(warped_bb_feat, f_dyn_state, block='cte')

                # Sample cte features
                f_cte = case["f_cte"].reshape(bs * self.n_objects * case["T"], -1)
                confi = case["confi"].reshape(bs * self.n_objects * case["T"], -1)
                #

                f_mu_cte, f_logvar_cte = self.linear_f_cte_post(f_cte).reshape(bs, self.n_objects, case["T"], -1).chunk(2, -1)
                f_cte_post = Normal(f_mu_cte, F.softplus(f_logvar_cte))
                f_cte = f_cte_post.rsample().reshape(bs * self.n_objects * case["T"], -1)
                # Option 2: Previous impl
                # f_mu_cte, f_logvar_cte = self.linear_f_cte_post(f_cte).reshape(bs, self.n_objects, 1, -1).chunk(2, -1)
                # f_cte_post = Normal(f_mu_cte, F.softplus(f_logvar_cte))
                # f_cte = f_cte_post.rsample()
                # f_cte = f_cte.repeat_interleave(case["T"], dim=2)
                # f_cte = f_cte.reshape(bs * self.n_objects * case["T"], self.feat_cte_dim)

                # Register statistics
                returned_post.append(f_cte_post)

                # Get full feature vector
                # f = torch.cat([ f_dyn_state,
                #                 f_cte], dim=-1) # Note: Do if f_dyn_state is used in the appearance
                f = f_cte

            # Get output. Spatial broadcast decoder

            dec_obj = self.image_decoder(f, block='to_x')
            if not self.bc_decoder:
                grid, area = self.spatial_tf(confi, pose)
                outs[case_name], out_shape = self.spatial_tf.warp_and_render(dec_obj, case["shape"], confi, grid)
            else:
                outs[case_name] = dec_obj * confi[..., None, None]
            # outs[case_name] = warp_affine(dec_obj, M_relocate, h=h, w=w)

            BBs[case_name] = dec_obj

        out_rec = outs["rec"]
        out_pred = outs["pred"]
        bb_rec = BBs["rec"]
        # bb_rec = BBs["pred"]

        # Test disentanglement - TO REVIEW
        # if random.random() < 0.1 or self.content is None:
        #     self.content = f_cte
        # f_cte = self.content

        ''' Returned variables '''
        returned_g = torch.cat([g, G_for_pred], dim=1) # Observations

        o_touple = (out_rec, out_pred, returned_g.reshape(-1, returned_g.size(-1)))
        o = [item.reshape(torch.Size([bs * self.n_objects, -1]) + item.size()[1:]) for item in o_touple]

        # Option 1: one object mapped to 0
        # shape_o0 = o[0].shape
        # o[0] = o[0].reshape(bs, self.n_objects, *o[0].shape[1:])
        # o[0][:,0] = o[0][:,0]*0
        # o[0] = o[0].reshape(*shape_o0)

        # Sum across objects. Note: Add Clamp or Sigmoid.
        o[:2] = [torch.clamp(torch.sum(item.reshape(bs, self.n_objects, *item.shape[1:]), dim=1), min=0, max=1) for item in o[:2]]
        # o[:2] = [torch.sum(item.reshape(bs, self.n_objects, *item.shape[1:]), dim=1) for item in o[:2]]
        # print(o[0].shape)

        # TODO: output = {}
        #  output['rec'] = o[0]
        #  output['pred'] = o[1]

        o.append(returned_post)
        o.append(A) # State transition matrix
        o.append(B) # Input matrix
        o.append(u.reshape(bs * self.n_objects, -1, u.shape[-1])) # Inputs
        o.append(u_dist.reshape(bs * self.n_objects, -1, u_dist.shape[-1])) # Input distribution
        o.append(g_for_koop.reshape(bs * self.n_objects, -1, g_for_koop.shape[-1])) # Observation propagated only with A
        o.append(fit_err) # Fit error g to G_for_pred

        # bb_rec = bb_rec.reshape(torch.Size([bs * self.n_objects, -1]) + bb_rec.size()[1:])
        # bb_rec =torch.sum(bb_rec, dim=2, keepdim=True)
        # o.append(bb_rec.reshape(bs, self.n_objects, *bb_rec.shape[1:])) # Motion field reconstruction

        # TODO: return in dictionary form
        return o