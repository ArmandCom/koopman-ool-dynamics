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

def local_geo(g, states, scaling_factor = 10):
    bs, T = states.shape[:2]
    permu = np.random.permutation(bs * (T))
    split_0 = permu[:bs * (T) // 2]
    split_1 = permu[bs * (T) // 2: 2*(bs * (T) // 2)]

    g = g.reshape(bs*T, 1, -1) # dim 1 stands for N_obj
    states = states.reshape(bs*T, 1, -1)

    dist_g = torch.mean((g[split_0] - g[split_1]) ** 2, dim=(1, 2))
    dist_s = torch.mean((states[split_0] - states[split_1]) ** 2, dim=(1, 2))

    return torch.abs(dist_g * scaling_factor - dist_s).mean()

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

def embedding_loss(output, target, epoch_iter, n_iters_start=3, lambd=0.3):
    # assert len(output) == 3 or len(output) == 5

    # TODO: implement KL-div with Pytorch lib
    #  passar distribucio i sample. Distribucio es calcula abans. Use rsample, no te lies.
    #  G**2 es el numero de slots. Seria com el numero de timesteps, suposo. Hauria de deixar el batch i prou.

    rec, pred, g, posteriors, A, B, u, qy, g_for_koop, fit_error = output[0], output[1], output[2], output[3], \
                                               output[4], output[5], output[6], \
                                                output[7], output[8], output[9]

    # Get bs, T, n_obj; and take order into account.
    bs = rec.shape[0]
    device = rec.device
    std_rec = .15

    # local_geo_loss = torch.zeros(1)
    if epoch_iter[0] < n_iters_start:
        lambd_lg = 1
        lambd_hrank = 0.1
        lambd_u = 1
    else:
        lambd_lg = 1
        lambd_hrank = 2
        lambd_u = 5

    '''Simple KL loss for reconstruction'''
    # a = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    # Shape latent: [bs, n_obj, T, dim]

    # B, ch, h, w
    # fg_dist = Normal(y_nobg, self.fg_sigma)
    # fg_likelihood = fg_dist.log_prob(x)
    # fg_likelihood = (fg_likelihood + (alpha_map + 1e-5).log())
    # bg_likelihood = (bg_likelihood + (1 - alpha_map + 1e-5).log())
    # # (B, 2, 3, H, W)
    # log_like = torch.stack((fg_likelihood, bg_likelihood), dim=1)
    # # (B, 3, H, W)
    # log_like = torch.logsumexp(log_like, dim=1)
    # # (B,)
    # log_like = log_like.flatten(start_dim=1).sum(1)

    '''Rec and pred losses'''
    # Shape latent: [bs, T-n_timesteps+1, 1, 64, 64]
    # TODO: Log_probs
    rec_distr = Normal(rec, std_rec)
    logprob_rec = rec_distr.log_prob(target[:, -rec.shape[1]:])\
        .flatten(start_dim=1).sum(1)

    pred_distr = Normal(pred, std_rec)
    logprob_pred = pred_distr.log_prob(target[:, -pred.shape[1]:])\
        .flatten(start_dim=1).sum(1) # TODO: Check if correct.


    # kl_z_pres = kl_divergence_bern_bern(z_pres_logits, self.prior_z_pres_prob)

    posteriors = posteriors[:-1] # If only rec
    prior = Normal(torch.zeros(1).to(device), torch.ones(1).to(device))
    kl_loss = torch.stack([kl_divergence(post, prior).flatten(start_dim=1).sum(1) for post in posteriors]).sum(0)

    nelbo = (kl_loss
             - logprob_rec
             # - logprob_pred
             ).mean()

    '''Observation space likelihood loss'''
    # T = (g.shape[1] -1)//2
    # g_mse_loss = mse_loss(g[:, 1:T+1], g[:, T+1:]).sum(dim=-1).mean()
    # g = g[:, :T]

    '''Low rank G'''
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
    h_rank_loss = (l1_loss(A, torch.zeros_like(A)).sum(-1).sum(-1).mean()
                           + l1_loss(B, torch.zeros_like(B)).sum(-1).sum(-1).mean())
    h_rank_loss = lambd_hrank * h_rank_loss

    '''Input KL div.'''
    # KL loss for gumbel-softmax. We should use increasing temperature for softmax. Check SPACE pres variable.

    '''Input sparsity loss
        u: [bs, T, feat_dim]
    '''
    up_bound = 0.2
    # l1_u_sparse = F.relu(l1_loss(u, torch.zeros_like(u)).mean() - up_bound)
    l1_u_diff_sparse = F.relu(l1_loss(u[:, :-1] - u[:, 1:], torch.zeros_like(u[:, 1:])).mean() - up_bound)
    l1_u = lambd_u * l1_u_diff_sparse


    '''Local geometry loss'''
    g = g[:, :rec.shape[1]]
    local_geo_loss = lambd_lg * local_geo(g, target[:, -g.shape[1]:])

    '''Total Loss'''
    loss = (  nelbo
            # + h_rank_loss
            # + l1_u
            # + fit_error
            # + local_geo_loss
            )

    rec_mse = mse_loss(rec, target[:, -rec.shape[1]:]).reshape(bs * rec.shape[1], -1).flatten(start_dim=1).sum(1).mean()
    pred_mse = mse_loss(pred, target[:, -pred.shape[1]:]).reshape(bs * pred.shape[1], -1).flatten(start_dim=1).sum(1).mean()

    return loss, {
              'Rec mse':rec_mse,
              'Pred mse':pred_mse,
              'KL Loss':kl_loss.mean(),
              'Rec llik':logprob_rec.mean(),
              'Pred llik':logprob_pred.mean(),
              'H rank Loss':h_rank_loss,
              #'A diag loss':loss_A,
              # #'G Pred Loss':g_mse_loss,
              # 'G Pred Loss':fit_error,
              # 'Local geo Loss':local_geo_loss,
              'l1_u':l1_u,
              }