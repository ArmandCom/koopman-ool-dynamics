import torch
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal, kl_divergence
from utils.util import linear_annealing

def embedding_loss(output, target, epoch_iter, n_epochs_start=0, lambd=0.3):
    # assert len(output) == 3 or len(output) == 5

    # TODO: REC COM F-1(F(F-1(F(S)))
    rec_ori = output["rec_ori"]
    pred = output["pred"]

    g = output["obs_rec_pred"]
    posteriors = output["gauss_post"]
    A = output["A"]
    B = output["B"]

    # Get bs, T, n_obj; and take order into account.
    bs = rec_ori.shape[0]
    T = rec_ori.shape[1]
    device = rec_ori.device
    std_rec = .15

    # local_geo_loss = torch.zeros(1)
    # Note: Let's see how it does with missing data
    # for k in ["rec", "pred", "pred_roll", "rec_ori"]:
    #     if output[k] is not None:
    #         output[k] = crop_top_left_keepdim(output[k], 96, None)

    lambd_fit_error = 0.0
    lambd_hrank = 0.0
    lambd_rec = 1
    lambd_pred = 0.0
    prior_pres_prob = 0
    prior_g_mask_prob = 0.0
    lambd_u = 0.0
    lambd_I = 0.0
    lambd_AE = 0.0
    lambd_u_rec = 0.0
    prior_coll_prob = 0.0
    lambd_sel = 0.0
    lambd_local_geo = 0.0
    # prior_pres_prob = linear_annealing(rec.device, epoch_iter[1], start_step=2000, end_step=40000, start_value=0.4, end_value=0.2)
    # prior_coll_prob = linear_annealing(rec.device, epoch_iter[1], start_step=2000, end_step=40000, start_value=0.1, end_value=0.01)
    # lambd_u_rec = linear_annealing(rec.device, epoch_iter[1], start_step=4000, end_step=40000, start_value=0, end_value=50)
    lambd_hrank = linear_annealing(device, epoch_iter[1], start_step=5000, end_step=10000, start_value=1, end_value=10)

    # Note: Mov mnist
    #TODO: Limit lambd_AE and fit error for missing case. And L1 instead of L2
    lambd_AE = linear_annealing(device, epoch_iter[1], start_step=4000, end_step=15000, start_value=0.1, end_value=10)
    lambd_fit_error = linear_annealing(device, epoch_iter[1], start_step=10000, end_step=20000, start_value=0.1, end_value=5)
    lambd_pred = linear_annealing(device, epoch_iter[1], start_step=4000, end_step=10000, start_value=0.2, end_value=1.3)
    # lambd_local_geo = linear_annealing(device, epoch_iter[1], start_step=2000, end_step=10000, start_value=1, end_value=10)
    lambd_u = linear_annealing(device, epoch_iter[1], start_step=4000, end_step=10000, start_value=0, end_value=10)

    # Note: Bouncing balls
    # lambd_AE = linear_annealing(device, epoch_iter[1], start_step=15000, end_step=20000, start_value=0, end_value=30)
    # lambd_local_geo = linear_annealing(device, epoch_iter[1], start_step=15000, end_step=20000, start_value=0, end_value=20)
    # lambd_pred = linear_annealing(device, epoch_iter[1], start_step=15000, end_step=20000, start_value=0.1, end_value=1)
    # lambd_u = linear_annealing(device, epoch_iter[1], start_step=4000, end_step=10000, start_value=0, end_value=10)
    # lambd_sel = linear_annealing(device, epoch_iter[1], start_step=10000, end_step=10000, start_value=0, end_value=10)

    '''Rec and pred losses'''
    # Shape latent: [bs, T-n_timesteps+1, 1, 64, 64]

    logprob_rec = 0.0
    if output["rec"] is not None:
        rec = output["rec"]
        rec_distr = Normal(rec, std_rec)
        logprob_rec = rec_distr.log_prob(target[:, -rec.shape[1]:]) \
            .flatten(start_dim=1).sum(1)

    logprob_pred = 0.0
    # n_preds = 5
    n_preds = pred.shape[1]
    pred_distr = Normal(pred[:, -n_preds:], std_rec)
    logprob_pred = pred_distr.log_prob(target[:, -n_preds:]) \
        .flatten(start_dim=1).sum(1)

    logprob_rec_ori = 0.0
    if output["rec_ori"] is not None:
        rec_ori = output["rec_ori"]
        rec_ori_distr = Normal(rec_ori, std_rec)
        logprob_rec_ori = rec_ori_distr.log_prob(target[:, -rec_ori.shape[1]:]) \
            .flatten(start_dim=1).sum(1)

    logprob_pred_roll = 0.0
    if output["pred_roll"] is not None:
        pred_roll = output["pred_roll"]
        pred_roll_distr = Normal(pred_roll, std_rec)
        logprob_pred_roll = pred_roll_distr.log_prob(target[:, -pred_roll.shape[1]:]) \
            .flatten(start_dim=1).sum(1)

    kl_bern_loss = 0.0
    '''G composition bernoulli KL div loss'''
    # if "g_bern_logit" == any(output.keys()):
    if output["g_bern_logit"] is not None:
        g_mask_logit = output["g_bern_logit"] # TODO: Check shape
        kl_g_mask_loss = kl_divergence_bern_bern(g_mask_logit, prior_pres_prob=torch.FloatTensor([prior_g_mask_prob]).to(g_mask_logit.device)).sum()
        kl_bern_loss = kl_bern_loss + kl_g_mask_loss


    '''Interaction bernoulli KL div loss'''
    # if "u_bern_logit" == any(output.keys()):
    if output["sel_bern_logit"] is not None and prior_coll_prob != 0:
        sel_logit = output["sel_bern_logit"]
        print(sel_logit.shape)
        exit('check that sel_logit last dimension is the number of objects')
        kl_sel_loss = kl_divergence_bern_bern(sel_logit, prior_pres_prob=torch.FloatTensor([prior_coll_prob/sel_logit.shape[-1]]).to(sel_logit.device)).sum()
        kl_bern_loss = kl_bern_loss + kl_sel_loss

    '''Input bernoulli KL div loss'''
    # if "u_bern_logit" == any(output.keys()):
    l1_u = 0.0
    if output["u_bern_logit"] is not None and prior_pres_prob != 0:
        u_logit = output["u_bern_logit"]
        kl_u_loss = kl_divergence_bern_bern(u_logit, prior_pres_prob=torch.FloatTensor([prior_pres_prob/u_logit.shape[-1]]).to(u_logit.device)).sum()
        # We normalize the probability by the number of possible activations.
        kl_bern_loss = kl_bern_loss + kl_u_loss
        l1_u = torch.tensor(0.0).to(device)
    elif lambd_u != 0 and output["u"] is not None:
        '''Input sparsity loss
        u: [bs, T, feat_dim]
        '''
        u = output["u"]
        up_bound = 0.2
        N_elem = u.shape[0]
        l1_u_sparse = F.relu(l1_loss(u, torch.zeros_like(u)).mean() - up_bound)
        # l1_u_diff_sparse = F.relu(l1_loss(u[:, :-1] - u[:, 1:], torch.zeros_like(u[:, 1:])).mean() - up_bound) * u.shape[0] * u.shape[1]
        l1_u = lambd_u * l1_u_sparse * N_elem
        # l1_u = lambd_u * l1_loss(u, torch.zeros_like(u)).mean()
    else:
        l1_u = torch.tensor(0.0).to(device)

    '''Gaussian vectors KL div loss'''
    posteriors = posteriors[:1] # If only rec
    prior = Normal(torch.zeros(1).to(device), torch.ones(1).to(device))
    kl_loss = torch.stack([kl_divergence(post, prior).flatten(start_dim=1).sum(1) for post in posteriors]).sum(0)

    nelbo = (kl_loss
             + kl_bern_loss
             - logprob_rec_ori * lambd_rec
             - logprob_rec * lambd_rec
             - logprob_pred     * lambd_pred
             - logprob_pred_roll * lambd_pred
             ).mean()

    AE_error = torch.tensor(0.0).to(device)
    if output["dyn_features"] is not None and lambd_AE != 0:
        dyn_feat = output["dyn_features"]
        rec_f_dyn = dyn_feat["rec"]
        pred_roll_f_dyn = dyn_feat["pred_roll"]
        pred_f_dyn = dyn_feat["pred"]

        T_pred_feat, _ = pred_roll_f_dyn .shape[1], pred_roll_f_dyn .shape[-1]
        # T_rec_feat, _ = rec_f_dyn .shape[-2], rec_f_dyn .shape[-1]

        # With reconstruction from original.
        ori_f_dyn = dyn_feat["rec_ori"]
        # T_ori_feat, dyn_dim = ori_f_dyn.shape[-2], ori_f_dyn.shape[-1]
        # AE_error = AE_error + lambd_AE * mse_loss(ori_f_dyn[:, -T_rec_feat:], rec_f_dyn).sum()
        # AE_error = AE_error + lambd_AE * mse_loss(ori_f_dyn[:, -T_pred_feat:], pred_f_dyn).sum()

        # With backwards prediction
        # pred_rev_f_dyn = dyn_feat["pred_rev"]
        # T_pred_rev_feat, _ = pred_rev_f_dyn .shape[-2], pred_rev_f_dyn .shape[-1]
        # AE_error = AE_error + lambd_AE * mse_loss(ori_f_dyn[:, -T_pred_rev_feat:], pred_rev_f_dyn).sum()

        weight = (torch.arange(pred_roll_f_dyn.shape[1])[None, :, None]*2 / pred_roll_f_dyn.shape[1]).float().to(device)

        AE_error = AE_error + lambd_AE * l1_loss(ori_f_dyn[:, -(pred_f_dyn.shape[1]-1):], pred_f_dyn[:, :-1]).sum()
        AE_error = AE_error + lambd_AE * l1_loss(ori_f_dyn[:, -(pred_roll_f_dyn.shape[1]):], pred_roll_f_dyn).sum() #*weight
        AE_error = AE_error + lambd_AE * l1_loss(ori_f_dyn[:, -(rec_f_dyn.shape[1]):], rec_f_dyn).sum()

        AE_error = torch.clamp(AE_error, max=10000)
        # TODO: add real autoencoding (input state to output state)

        # AE_distr = Normal(pred_f_dyn, 0.01)
        # AE_error = -lambd_AE * AE_distr.log_prob(rec_f_dyn[:, -T_pred_feat:]) \
        # .flatten(start_dim=1).sum()

    '''LSQ fit loss'''
    # Note: only has correspondence if last frames of pred correspond to last frames of rec
    fit_error = torch.tensor(0.0).to(device)
    if lambd_fit_error != 0:
        g_rec, g_pred = g[:, :T], g[:, T:]
        T_min = min(T, g_pred.shape[1])

        # Option 1: Long term
        # fit_error = fit_error + lambd_fit_error * mse_loss(g_rec[:, -T_min:], g_pred[:, -T_min:]).sum()

        # Option 2: Short term
        g_pred_1step = output['g_pred_1step']
        fit_error = fit_error + lambd_fit_error * mse_loss(g_pred_1step[:, :-1], g_rec[:, -(g_pred_1step.shape[1]-1):]).sum()

        g_pred_2step = output['g_pred_2step']
        fit_error = fit_error + lambd_fit_error * mse_loss(g_pred_2step[:, :-2], g_rec[:, -(g_pred_1step.shape[1]-2):]).sum()

    mse_u_loss = torch.tensor(0.0).to(device)
    if output["u_rec"] is not None and output["u"] is not None and lambd_u_rec!=0:
        u_rec, u = output["u_rec"], output["u"]
        mse_u_loss = mse_u_loss + lambd_u_rec *  l1_loss(u_rec[:, -u.shape[1]:], u).sum()

    local_geo_loss = torch.tensor(0.0).to(device)
    if lambd_local_geo!=0:
        local_geo_loss = local_geo_loss + lambd_local_geo *  local_geo(g[:, :T], dyn_feat["rec_ori"], rec_ori)
    '''Low rank G'''
    # Option 1:
    # T = g_for_koop.shape[1]
    # n_timesteps = g_for_koop.shape[-1]
    # g_for_koop = g_for_koop.permute(0, 2, 1, 3).reshape(-1, T-1, n_timesteps)
    # h_rank_loss = 0
    # reg_mask = torch.zeros_like(g_for_koop[..., :n_timesteps, :])
    # ids = torch.arange(0, reg_mask.shape[-1])
    # reg_mask[..., ids, ids] = 0.01
    # for t in range(T-1-n_timesteps):
    #     logdet_H = torch.slogdet(g_for_koop[..., t:t+n_timesteps, :] + reg_mask)
    #     h_rank_loss = h_rank_loss + .01*(logdet_H[1]).mean()

    # Option 2:
    h_rank_loss = torch.tensor(0.0).to(device)
    if lambd_hrank != 0:
        # h_rank_loss = (l1_loss(A, torch.zeros_like(A)).sum(-1).sum(-1).sum()
                       # + l1_loss(B, torch.zeros_like(B)).sum(-1).sum(-1).mean()
                       # )
        # h_rank_loss = (l1_loss(A, torch.zeros_like(A)).sum(-1).sum(-1).sum()
        #                + l1_loss(B, torch.zeros_like(B)).sum(-1).sum(-1).sum())

        h_rank_loss = torch.norm(A, p='nuc', dim=[0, 1]).sum()
        h_rank_loss = lambd_hrank * h_rank_loss

    '''Local geometry loss'''
    # g = g[:, :rec.shape[1]]
    # local_geo_loss = lambd_lg * local_geo(g, target[:, -g.shape[1]:])

    '''Sel L1'''
    l1_sel = torch.tensor(0.0).to(device)
    if lambd_sel != 0 and output["sel"] is not None:
        '''Input sparsity loss
            u: [bs, T, feat_dim]
        '''
        sel = output["sel"]
        up_bound = 0.3
        N_elem = sel.shape[0]
        l1_sel = F.relu(l1_loss(u, torch.zeros_like(u)).mean() - up_bound)
        # l1_u_diff_sparse = F.relu(l1_loss(u[:, :-1] - u[:, 1:], torch.zeros_like(u[:, 1:])).mean() - up_bound) * u.shape[0] * u.shape[1]
        l1_sel = lambd_sel * l1_sel * N_elem
        # l1_u = lambd_u * l1_loss(u, torch.zeros_like(u)).mean()

    '''Total Loss'''
    loss = (  nelbo
              # + h_rank_loss
              + l1_u
              # + local_geo_loss
              + fit_error
              + AE_error
              # + mse_u_loss
              )


    rec_mse = mse_loss(rec_ori, target[:, -rec_ori.shape[1]:]).mean(1).reshape(bs, -1).flatten(start_dim=1).sum(1).mean()
    pred_mse = mse_loss(pred, target[:, -pred.shape[1]:]).mean(1).reshape(bs, -1).flatten(start_dim=1).sum(1).mean()

    dict_out = {
        'Rec mse':rec_mse,
        'Pred mse':pred_mse,
        'KL Loss':kl_loss.mean(),
        # 'Rec llik':logprob_rec.mean(),
        # 'Pred llik':logprob_pred.mean(),
        # 'Cycle consistency Loss':cyc_con_loss,
        # 'H rank Loss':h_rank_loss,
        'Fit error':fit_error,
        'AE error':AE_error,
        # 'G Pred Loss':fit_error,d
        'Local geo Loss':local_geo_loss,
        # 'l1_u':l1_u,
    }

    return loss, dict_out

def nll_loss(output, target):
    return F.nll_loss(output, target)

def mse_loss(output, target, reduction='none'):
    return F.mse_loss(output, target, reduction=reduction)

def l1_loss(output, target, reduction='none'):
    return F.l1_loss(output, target, reduction=reduction)

def local_geo(g, states, vid, scaling_factor = 0.1):
    bsO, T = states.shape[:2]
    permu = np.random.permutation(bsO * (T))
    split_0 = permu[:bsO * (T) // 2]
    split_1 = permu[bsO * (T) // 2: 2*(bsO * (T) // 2)]

    g = g.reshape(bsO*T, 1, -1) # dim 1 stands for N_obj
    states = states.reshape(bsO*T, 1, -1)

    O = bsO//vid.shape[0]
    vid = vid.repeat_interleave(O, dim=0).reshape(bsO*T, 1, -1)

    dist_g = torch.sum((g[split_0] - g[split_1]) ** 2, dim=-1)
    dist_s = torch.sum((states[split_0] - states[split_1]) ** 2, dim=-1)
    dist_vid = torch.sum((vid[split_0] - vid[split_1]) ** 2, dim=-1)
    return torch.abs(dist_g * scaling_factor - dist_s).sum() + torch.abs(dist_s * scaling_factor - dist_vid * 2).sum()

def diagonal_loss(M):
    assert M.shape[-1] == M.shape[-2]
    mask = torch.zeros_like(M)
    ids = torch.arange(0,mask.shape[-1])
    mask[..., ids, ids] = 1
    l2_off_diag_M = mse_loss(M * (1 - mask), torch.zeros_like(M)).sum(-1).sum(-1).sum(-1).mean()
    l1_diag_M = (M * mask).sum(-1).sum(-1).sum(-1).mean()
    M_loss = l2_off_diag_M + l1_diag_M
    # l1_M = l1_loss(M, mask).sum(-1).sum(-1).sum(-1).mean()
    return M_loss

def cycle_consistency_loss(A, B):

    loss_consist = 0.0
    K = A.shape[-1]

    for k in range(1,K+1):
        As1 = A[:,:k]
        Bs1 = B[:k,:]
        As2 = A[:k,:]
        Bs2 = B[:,:k]

        Ik = torch.eye(k).float().to(A.device)

        if k == 1:
            loss_consist = (torch.sum((torch.mm(Bs1, As1) - Ik)**2) + \
                            torch.sum((torch.mm(As2, Bs2) - Ik)**2) ) / (2.0*k)
        else:
            loss_consist += (torch.sum((torch.mm(Bs1, As1) - Ik)**2) + \
                             torch.sum((torch.mm(As2, Bs2)-  Ik)**2) ) / (2.0*k)

    #                Ik = torch.eye(K).float().to(device)
    #                loss_consist = (torch.sum( (torch.mm(A, B)-Ik )**2)**1 + \
    #                                         torch.sum( (torch.mm(B, A)-Ik)**2)**1 )

    return loss_consist

def kl_divergence_bern_bern(z_pres_logits, prior_pres_prob, eps=1e-15):
    """
    Compute kl divergence of two Bernoulli distributions
    :param z_pres_logits: (B, ...)
    :param prior_pres_prob: float
    :return: kl divergence, (B, ...)
    """
    z_pres_probs = torch.sigmoid(z_pres_logits)
    kl = z_pres_probs * (torch.log(z_pres_probs + eps) - torch.log(prior_pres_prob + eps)) + \
         (1 - z_pres_probs) * (torch.log(1 - z_pres_probs + eps) - torch.log(1 - prior_pres_prob + eps))

    return kl

def crop_top_left_keepdim(img, cropx, cropy):
    y, x = img.shape[-2:]
    mask = torch.ones_like(img)
    # mask[..., :, :(x - cropx)] = 0
    mask[..., :(y - cropx), :] = 0
    return img * mask