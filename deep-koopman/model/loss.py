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
    mask = torch.ones_like(M)
    ids = torch.arange(0,mask.shape[-1])
    mask[..., ids, ids] = 0
    l1_M = l1_loss(M*mask, torch.zeros_like(M)).sum(-1).sum(-1).sum(-1).mean()
    return l1_M

def embedding_loss(output, target, lambd=0.3):
    # assert len(output) == 3 or len(output) == 5

    rec, pred, g, mu, logvar, A, _, u = output[0], output[1], output[2], output[3], \
                                        output[4], output[5], output[6], output[7]

    kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
    # var = logvar
    # kl_loss = 0.5 * (torch.einsum('bii->b', var) + mu.pow(2).sum(-1) - torch.logdet(var).sum(-1)).mean()

    loss_diag_A = diagonal_loss(A)

    rec_loss = 10 * mse_loss(rec, target[:, -rec.shape[1]:])\
        .view(rec.size(0)*(rec.size(1)), -1).sum(dim=-1).mean()

    l1_u = l1_loss(u, torch.zeros_like(u)).sum(-1).sum(-1).mean()
    #TODO: loss for A matrix except diagonal.

    # pred_loss = torch.zeros(1)
    free_pred = 0
    if free_pred < 1:
        pred_loss = 10 * mse_loss(pred, target[:, -pred.shape[1]:]) \
            .view(pred.size(0) * (pred.size(1)), -1).sum(dim=-1).mean()
    else:
        pred_loss = 10 * mse_loss(pred[:, :-free_pred], target[:, -pred.shape[1]:-free_pred])\
            .view(pred.size(0)*(pred.size(1)-free_pred), -1).sum(dim=-1).mean()

    local_geo_loss = torch.zeros(1)
    # local_geo_loss = local_geo(g, target[:, -g.shape[1]:])

    loss = rec_loss + pred_loss + kl_loss + l1_u + loss_diag_A #+ lambd * local_geo_loss

    return loss, {'Rec Loss':rec_loss, 'Pred Loss':pred_loss,
                  'KL Loss':kl_loss, 'L1 U Loss':l1_u, 'A_diag_loss':loss_diag_A}


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