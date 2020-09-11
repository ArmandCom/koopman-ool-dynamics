import torch
import torch.nn.functional as F
import numpy as np

def nll_loss(output, target):
    return F.nll_loss(output, target)

def mse_loss(output, target):
    return F.mse_loss(output, target)

def l1_loss(output, target):
    return F.l1_loss(output, target)

def local_geo(g, states, scaling_factor = 10):
    bs, T = states.shape[:2] #TODO: reshape g from source
    permu = np.random.permutation(bs * (T))
    split_0 = permu[:bs * (T) // 2]
    split_1 = permu[bs * (T) // 2:]

    dist_g = torch.mean((g[split_0] - g[split_1]) ** 2, dim=(1, 2))
    dist_s = torch.mean((states[split_0] - states[split_1]) ** 2, dim=(1, 2))

    return torch.abs(dist_g * scaling_factor - dist_s).mean()

def embedding_loss(output, target, lambd=0.3):
    assert len(output) == 3
    rec, pred, g = output
    rec_loss, pred_loss, local_geo_loss = l1_loss(rec, target), l1_loss(pred, target[:, 1:]), local_geo(g, target)
    loss = rec_loss + pred_loss + lambd * local_geo_loss
    return loss, {'Rec Loss':rec_loss, 'Pred Loss':pred_loss, 'Local Geometry Loss':local_geo_loss}

# permu = np.random.permutation(bs * (T + 1))
# split_0 = permu[:bs * (T + 1) // 2]
# split_1 = permu[bs * (T + 1) // 2:]
# dist_g = torch.mean((g[split_0] - g[split_1]) ** 2, dim=(1, 2))
# dist_s = torch.mean((states_flat[split_0] - states_flat[split_1]) ** 2, dim=(1, 2))
# scaling_factor = 10
# loss_metric = torch.abs(dist_g * scaling_factor - dist_s).mean()
# loss_auto_encode = F.l1_loss(decode_s_for_ae, states[:, :T + 1].reshape(decode_s_for_ae.shape))
# loss_prediction = F.l1_loss(decode_s_for_pred, states[:, 1:].reshape(decode_s_for_pred.shape))
#
# loss = loss_auto_encode + loss_prediction + loss_metric * args.lambda_loss_metric