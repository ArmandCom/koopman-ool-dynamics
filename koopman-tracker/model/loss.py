import torch
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal, kl_divergence

def nll_loss(output, target):
    return F.nll_loss(output, target)

def mse_loss(output, target, reduction='none'):
    return F.mse_loss(output, target, reduction=reduction)

def l1_loss(output, target, reduction='none'):
    return F.l1_loss(output, target, reduction=reduction)

# def local_geo(g, states, scaling_factor = 10):
#     bs, T = states.shape[:2]
#     permu = np.random.permutation(bs * (T))
#     split_0 = permu[:bs * (T) // 2]
#     split_1 = permu[bs * (T) // 2: 2*(bs * (T) // 2)]
#
#     g = g.reshape(bs*T, 1, -1) # dim 1 stands for N_obj
#     states = states.reshape(bs*T, 1, -1)
#
#     dist_g = torch.mean((g[split_0] - g[split_1]) ** 2, dim=(1, 2))
#     dist_s = torch.mean((states[split_0] - states[split_1]) ** 2, dim=(1, 2))
#
#     return torch.abs(dist_g * scaling_factor - dist_s).mean()

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

def embedding_loss(output, target, epoch_iter, n_iters_start=0, lambd=0.3):
    # assert len(output) == 3 or len(output) == 5

    # TODO: implement KL-div with Pytorch lib
    #  passar distribucio i sample. Distribucio es calcula abans. Use rsample, no te lies.
    #  G**2 es el numero de slots. Seria com el numero de timesteps, suposo. Hauria de deixar el batch i prou.

    rec = output["rec"]
    pred = output["pred"]
    pred_rev = output["pred_rev"]

    g = output["obs_rec_pred"]
    posteriors = output["gauss_post"]
    A = output["A"]
    B = output["B"]
    u = output["u"]

    # Get bs, T, n_obj; and take order into account.
    bs = rec.shape[0]
    T = rec.shape[1]
    device = rec.device
    std_rec = .15

    # local_geo_loss = torch.zeros(1)
    if epoch_iter[0] < n_iters_start: #TODO: Add rec loss in G with spectralnorm. fit_error
        # lambd_lg = 1
        lambd_fit_error = 1
        lambd_hrank = 0.05
        lambd_rec = 1
        lambd_pred = 0.2
        prior_pres_prob = 0.05
        prior_g_mask_prob = 0.8
        lambd_u = 0.1
        lambd_I = 1
    else:
        # lambd_lg = 1
        lambd_fit_error = 1
        lambd_hrank = 0.2
        lambd_rec = .7
        lambd_pred = 1
        prior_pres_prob = 0.1
        prior_g_mask_prob = 0.4
        lambd_u = 1
        lambd_I = 1

    '''Rec and pred losses'''
    # Shape latent: [bs, T-n_timesteps+1, 1, 64, 64]

    rec_distr = Normal(rec, std_rec)
    logprob_rec = rec_distr.log_prob(target[:, -rec.shape[1]:])\
        .flatten(start_dim=1).sum(1)

    pred_distr = Normal(pred, std_rec)
    logprob_pred = pred_distr.log_prob(target[:, -pred.shape[1]:])\
        .flatten(start_dim=1).sum(1) # TODO: Check if correct.

    pred_rev_distr = Normal(pred_rev, std_rec)
    logprob_pred_rev = pred_rev_distr.log_prob(target[:, :pred_rev.shape[1]]) \
    .flatten(start_dim=1).sum(1) # TODO: Check if correct.

    kl_bern_loss = 0
    '''G composition bernoulli KL div loss'''
    # if "g_bern_logit" == any(output.keys()):
    if output["g_bern_logit"] is not None:
        g_mask_logit = output["g_bern_logit"] # TODO: Check shape
        kl_g_mask_loss = kl_divergence_bern_bern(g_mask_logit, prior_pres_prob=torch.FloatTensor([prior_g_mask_prob]).to(u_logit.device)).sum()
        kl_bern_loss = kl_bern_loss + kl_g_mask_loss

    '''Input bernoulli KL div loss'''
    # if "u_bern_logit" == any(output.keys()):
    if output["u_bern_logit"] is not None:
        u_logit = output["u_bern_logit"]
        kl_u_loss = kl_divergence_bern_bern(u_logit, prior_pres_prob=torch.FloatTensor([prior_pres_prob]).to(u_logit.device)).sum()
        kl_bern_loss = kl_bern_loss + kl_u_loss
        l1_u = 0.
    else:
        '''Input sparsity loss
        u: [bs, T, feat_dim]
        '''
        up_bound = 0.3
        N_elem = u.shape[0]
        l1_u_sparse = F.relu(l1_loss(u, torch.zeros_like(u)).mean() - up_bound)
        # l1_u_diff_sparse = F.relu(l1_loss(u[:, :-1] - u[:, 1:], torch.zeros_like(u[:, 1:])).mean() - up_bound) * u.shape[0] * u.shape[1]
        l1_u = lambd_u * l1_u_sparse * N_elem
        # l1_u = lambd_u * l1_loss(u, torch.zeros_like(u)).mean()

    '''Gaussian vectors KL div loss'''
    posteriors = posteriors[:-1] # If only rec
    prior = Normal(torch.zeros(1).to(device), torch.ones(1).to(device))
    kl_loss = torch.stack([kl_divergence(post, prior).flatten(start_dim=1).sum(1) for post in posteriors]).sum(0)

    nelbo = (kl_loss
             + kl_bern_loss
             - logprob_rec * lambd_rec
             - logprob_pred     * lambd_pred #TODO: There's a mismatch here?
             - logprob_pred_rev * lambd_pred
             ).mean()

    '''Cycle consistency'''
    if output["Ainv"] is not None:
        A_inv = output["Ainv"]
        cyc_conc_loss = cycle_consistency_loss(A, A_inv)

    '''LSQ fit loss'''
    # Note: only has correspondence if last frames of pred correspond to last frames of rec
    g_rec, g_pred = g[:, :T], g[:, T:]
    T_min = min(T, g_pred.shape[1])
    fit_error = lambd_fit_error * mse_loss(g_rec[:, -T_min:], g_pred[:, -T_min:]).sum()

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
    h_rank_loss = (l1_loss(A, torch.zeros_like(A)).sum(-1).sum(-1).sum()
                           # + l1_loss(B, torch.zeros_like(B)).sum(-1).sum(-1).mean()
                   )
    # h_rank_loss = (l1_loss(A, torch.zeros_like(A)).sum(-1).sum(-1).sum()
    #                + l1_loss(B, torch.zeros_like(B)).sum(-1).sum(-1).sum())
    h_rank_loss = lambd_hrank * h_rank_loss

    '''Input KL div.'''
    # KL loss for gumbel-softmax. We should use increasing temperature for softmax. Check SPACE pres variable.


    '''Local geometry loss'''
    # g = g[:, :rec.shape[1]]
    # local_geo_loss = lambd_lg * local_geo(g, target[:, -g.shape[1]:])

    '''Total Loss'''
    loss = (  nelbo
            + h_rank_loss
            + l1_u
            + fit_error
            + cyc_cons_loss
            # + local_geo_loss
            )

    rec_mse = mse_loss(rec, target[:, -rec.shape[1]:]).mean(1).reshape(bs, -1).flatten(start_dim=1).sum(1).mean()
    pred_mse = mse_loss(pred, target[:, -pred.shape[1]:]).mean(1).reshape(bs, -1).flatten(start_dim=1).sum(1).mean()

    return loss, {
              'Rec mse':rec_mse,
              'Pred mse':pred_mse,
              'KL Loss':kl_loss.mean(),
              # 'Rec llik':logprob_rec.mean(),
              # 'Pred llik':logprob_pred.mean(),
              'Cycle consistency Loss':cyc_cons_loss,
              'H rank Loss':h_rank_loss,
              'Fit error':fit_error,
              # 'G Pred Loss':fit_error,
              # 'Local geo Loss':local_geo_loss,
              # 'l1_u':l1_u,
              }