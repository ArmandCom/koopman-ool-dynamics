import torch
import torch.nn.functional as F
import numpy as np

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

def logvar_to_matrix_var(logvar):
    var = torch.exp(logvar)
    var_mat = torch.diag_embed(var)
    return var_mat

def embedding_loss_bidir(output, target, lambd=0.3):
    # assert len(output) == 3 or len(output) == 5

    rec, pred, g, mu, logvar, A, _, u, g_rev = output[0], output[1], output[2], output[3], \
                                        output[4], output[5], output[6], output[7], output[8]

    kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())[:, 0].sum(dim=-1).mean() #.sum(dim=1)

    # var = logvar_to_matrix_var(logvar)
    # kl_loss = 0
    # mu_t, var_t = mu[:, 0], var[:, 0]
    # for _ in range(pred.shape[2]):
    #     mu_t, var_t = torch.bmm(mu_t, A), torch.bmm(torch.bmm(A.permute(0,1,3,2) ,var_t), A)
    #     kl_loss = kl_loss + \
    #               0.5 * (torch.einsum('bii->b', var_t) + mu_t.pow(2).sum(-1) -
    #                      torch.logdet(var_t).sum(-1)).mean()

    # T = (g.shape[1] -1)//2
    # g_mse_loss = mse_loss(g[:, 1:T+1], g[:, T+1:]).sum(dim=-1).mean()
    # g = g[:, :T]

    # loss_A = diagonal_loss(A)
    # bs = rec.shape[0] // 2
    rec = rec.reshape(-1, 2, *rec.shape[1:])
    rec_loss_rev = 10 * mse_loss(rec[:,1], torch.flip(target[:, :rec.shape[2]], dims=[1]))\
        .view(rec.size(0)*(rec.size(2)), -1).sum(dim=-1).mean()
    rec_loss = 10 * mse_loss(rec[:, 0], target[:, -rec.shape[2]:])\
        .view(rec.size(0)*(rec.size(2)), -1).sum(dim=-1).mean()

    # l1_u = l1_loss(u, torch.zeros_like(u)).sum(-1).sum(-1).mean()
    #TODO: loss for A matrix except diagonal.

    # pred_loss = torch.zeros(1)
    free_pred = 0
    pred = pred.reshape(-1, 2, *pred.shape[1:])
    if free_pred < 1:
        pred_loss_rev = 10 * mse_loss(pred[:,1], torch.flip(target[:, :pred.shape[2]], dims=[1])) \
            .view(pred.size(0)*(pred.size(2)), -1).sum(dim=-1).mean()
        pred_loss = 10 * mse_loss(pred[:,0], target[:, -pred.shape[2]:]) \
            .view(pred.size(0) * (pred.size(2)), -1).sum(dim=-1).mean()
    # else:
    #     pred_loss_rev = 10 * mse_loss(pred[bs:, :-free_pred], torch.flip(target[:, :pred.shape[1]-free_pred], dims=[1])) \
    #         .view(pred.size(0)*(pred.size(1)), -1).sum(dim=-1).mean()
    #     pred_loss = 10 * mse_loss(pred[:bs, :-free_pred], target[:, -pred.shape[1]:-free_pred])\
    #         .view(pred.size(0)*(pred.size(1)-free_pred), -1).sum(dim=-1).mean()

    local_geo_loss = torch.zeros(1)
    # local_geo_loss = local_geo(g, target[:, -g.shape[1]:])

    loss = rec_loss + pred_loss + kl_loss #+ rec_loss_rev + pred_loss_rev #g_mse_loss + lambd * local_geo_loss #+ loss_diag_A #+ l1_u + lambd * local_geo_loss
    #, 'L1 U Loss':l1_u, 'A_diag_loss':loss_diag_A

    return loss, {'Rec Loss':rec_loss, 'Pred Loss':pred_loss, 'Rec rev Loss':rec_loss_rev, 'Pred rev Loss':pred_loss_rev, #'A diag loss':loss_A, #'G Pred Loss':g_mse_loss,
                  'KL Loss':kl_loss}#, 'Local geo Loss':local_geo_loss}

def embedding_loss(output, target, epoch_iter, n_iters_start=3, lambd=0.3):
    # assert len(output) == 3 or len(output) == 5

    rec, pred, g, mu, logvar, A, B, u, qy, g_for_koop, fit_error = output[0], output[1], output[2], output[3], \
                                               output[4], output[5], output[6], \
                                                output[7], output[8], output[9], output[10]

    '''Simple KL loss for reconstruction'''
    kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()#.sum(dim=1)
    '''Discrete input'''
    # Option 1:
    # log_qy = torch.log(qy+1e-20)
    # g = Variable(torch.log(torch.Tensor([1.0/2])).cuda())
    # kl_loss_gumb = torch.sum(qy*(log_qy - g),dim=-1).mean()
    # Option 2:
    # kl_loss_gumb = - torch.sum(qy*torch.log(qy + 1e-6))
    # Option 3:
    # n_cat = qy.shape[-1]
    # log_ratio = torch.log(qy * n_cat + 1e-10)
    # # TODO: reshape for objects and sum them out?
    # # TODO: are we imposing all classes have the same probability?
    # # TODO: Take the time to understand Gumbel
    # kl_loss_gumb = torch.sum(qy * log_ratio, dim=-1).mean()
    # kl_loss = kl_loss + kl_loss_gumb

    ''' KL loss for prediction:
        Computation of the KL divergence for non-diagonal Sigmas'''
    # mu = mu.reshape(*mu.shape[:-1], A.shape[-3], -1)
    # logvar = logvar.reshape(*logvar.shape[:-1], A.shape[-3], -1)
    # n_objects = mu.shape[1]
    # var = logvar_to_matrix_var(logvar)
    # mu, var = mu.reshape(-1, *mu.shape[2:]), var.reshape(-1, *var.shape[2:])

    # kl_loss = 0
    # mu_t, var_t = mu[:, 0], var[:, 0]
    # for _ in range(pred.shape[2]):
    #     mu_t, var_t = torch.bmm(mu_t, A), \
    #                   torch.bmm(torch.bmm(A.permute(0,1,3,2) ,var_t), A)
    #     kl_loss = kl_loss + \
    #               0.5 * (torch.einsum('bii->b', var_t) + mu_t.pow(2).sum(-1) -
    #                      torch.logdet(var_t).sum(-1))
    # kl_loss = kl_loss.reshape(-1, n_objects, *kl_loss.shape[1:])

    '''Observation space likelihood loss'''
    # T = (g.shape[1] -1)//2
    # g_mse_loss = mse_loss(g[:, 1:T+1], g[:, T+1:]).sum(dim=-1).mean()
    # g = g[:, :T]

    '''Desired A loss'''
    # loss_A = diagonal_loss(A)

    '''Low rank G'''
    # Option 1:
    if epoch_iter[0] < n_iters_start:
        lambd_1 = 0.1
    else:
        lambd_1 = 2

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
    h_rank_loss = lambd_1 * h_rank_loss
    '''Input sparsity loss
        u: [bs, T, feat_dim]
    '''
    # Option 1: If activations are not binary,
    #  penalize them being activated + being close to 0.5
    # l1_u = 2 * (l1_loss(u, torch.zeros_like(u)).sum(-1).sum(-1).mean() -
    #        mse_loss(u, torch.ones_like(u)*0.5).sum(-1).sum(-1).mean())
    # Option 2: Penalize activations
    if epoch_iter[0] < n_iters_start:
        lambd_2 = .1
    else:
        lambd_2 = 1
    l1_u = l1_loss(u, torch.zeros_like(u)).sum(-1).sum(-1).mean()
    # Option 3: Penalize if a specific dimension has a single activation in it,
    #  but it doesn't matter how many timesteps.
    # u_max, _ = torch.max(u, dim=1)
    # l1_u = 50 * l1_loss(u_max, torch.zeros_like(u_max)).sum(-1).mean()
    # Option 4: Sequential Temporal, and feature sparsity
    # l1_u_T = -((u[:, 1:] + u[:, :-1]) * l1_loss(u[:, 1:], u[:, :-1])).sum(-1).sum(-1).mean()
    # u_ones = torch.ones_like(u)
    # l1_u_T_flat = -((u_ones[:, 1:] - (u[:, 1:] + u[:, :-1])) * (u_ones[:, 1:] - l1_loss(u[:, 1:], u[:, :-1]))).sum(-1).sum(-1).mean()

    # l1_u_F = -(((u[:, :, 1:] + u[:, :, :-1]) * l1_loss(u[:, :, 1:], u[:, :, :-1]))**2).sum(-1).sum(-1).mean()

    l1_u = lambd_2 * (l1_u) # + l1_u_T + l1_u_T_flat + l1_u_F

    '''Rec and pred losses'''
    rec_loss = 10 * mse_loss(rec, target[:, -rec.shape[1]:]) \
        .view(rec.size(0)*(rec.size(1)), -1).sum(dim=-1).mean()
    # pred_loss = torch.zeros(1)

    # if epoch_iter[0] < n_iters_start:
    #     lambd_pred = 1
    # else:
    #     lambd_pred = 0
    free_pred = 0
    if free_pred < 1:
        pred_loss = 10 * mse_loss(pred, target[:, -pred.shape[1]:]) \
            .view(pred.size(0) * (pred.size(1)), -1).sum(dim=-1).mean()
    else:
        pred_loss = 10 * mse_loss(pred[:, :-free_pred], target[:, -pred.shape[1]:-free_pred]) \
            .view(pred.size(0)*(pred.size(1)-free_pred), -1).sum(dim=-1).mean()

    '''Local geometry loss'''
    # local_geo_loss = torch.zeros(1)
    if epoch_iter[0] < n_iters_start:
        lambd_3 = 0
    else:
        lambd_3 = 0
    g = g[:, :rec.shape[1]]
    local_geo_loss = lambd_3 * local_geo(g, target[:, -g.shape[1]:])


    if epoch_iter[0] < n_iters_start:
        lambd_4 = 0
    else:
        lambd_4 = 0
    fit_error = lambd_4 * fit_error

    '''Total Loss'''
    loss = (  rec_loss
            + pred_loss
            + kl_loss
            # + h_rank_loss
            + l1_u
            # + fit_error
            # + local_geo_loss
            )

    return loss, {'Rec Loss':rec_loss,
                  'Pred Loss':pred_loss,
                  'H rank Loss':h_rank_loss,
                  #'A diag loss':loss_A,
                  # #'G Pred Loss':g_mse_loss,
                  # 'G Pred Loss':fit_error,
                  # 'Local geo Loss':local_geo_loss,
                  'KL Loss':kl_loss,
                  'l1_u':l1_u,
                  }

# def explicit_embedding_loss(output, target, lambd=0.3):
#     assert len(output) == 4
#
#     # chequear input range
#     rec, pred, g, kl_loss = output
#
#     # reconstruct
#     rec = rec.view(*rec.shape[:2], -1)
#     size = rec.size(-1)
#     target_rec = target[:, -rec.shape[1]:]
#     log_p_x_g_z = -F.binary_cross_entropy_with_logits(rec, target_rec.view(*rec.shape)) * size
#
#     free_pred = 0
#     rec_loss = -log_p_x_g_z
#     pred_loss = torch.zeros(1)
#     # rec_loss = 10 * mse_loss(rec, target[:, -rec.shape[1]:])\
#     #     .view(rec.size(0)*(rec.size(1)), -1).sum(dim=-1).mean()
#     # # # pred_loss = 10 * mse_loss(pred[:, :-free_pred], target[:, -pred.shape[1]:-free_pred])\
#     # # #     .view(pred.size(0)*(pred.size(1)-free_pred), -1).sum(dim=-1).mean()
#     # log_p_x_g_z = -rec_loss #- 0 * pred_loss
#
#     elbo = log_p_x_g_z - kl_loss
#     # local_geo_loss = local_geo(g, target[:, -g.shape[1]:])
#
#     loss = -elbo #+ lambd * local_geo_loss
#
#     return loss, {'Rec Loss':rec_loss, 'Pred Loss':pred_loss, 'KL Loss':kl_loss} #'Local Geometry Loss':local_geo_loss,