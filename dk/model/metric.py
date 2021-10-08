import torch
import numpy as np
import frechet_video_distance as fvd
from skimage.measure import compare_ssim as ssim
from PerceptualSimilarity import lpips
import tensorflow as tf
import torch.nn as nn

# def mse_mae_ssim(predictions, target):
#     batch_size, target_length, loader_length = None, None, None
#
#     total_mse, total_mae,total_ssim,total_bce = 0,0,0,0
#     mse_batch = np.mean((predictions-target)**2 , axis=(0,1,2)).sum()
#     mae_batch = np.mean(np.abs(predictions-target) ,  axis=(0,1,2)).sum()
#     total_mse += mse_batch
#     total_mae += mae_batch
#     for a in range(0,target.shape[0]):
#         for b in range(0,target.shape[1]):
#             total_ssim += ssim(target[a,b,0,], predictions[a,b,0,]) / (target.shape[0]*target.shape[1])
#     cross_entropy = -target*np.log(predictions) - (1-target) * np.log(1-predictions)
#     cross_entropy = cross_entropy.sum()
#     cross_entropy = cross_entropy / (batch_size*target_length)
#     total_bce +=  cross_entropy
#     print('eval mse ', total_mse/loader_length,  ' eval mae ', total_mae/loader_length,' eval ssim ',total_ssim/loader_length, ' eval bce ', total_bce/loader_length)
#     return total_mse/loader_length,  total_mae/loader_length, total_ssim/loader_length
bce_loss = nn.BCELoss()
mse_loss = nn.MSELoss()
def mse(predictions, target):
    predictions = predictions.cpu().numpy()
    target = target.cpu().numpy()
    mse_batch = np.mean((predictions-target)**2 , axis=(0,1)).sum()
    return mse_batch

def mae(predictions, target):
    predictions = predictions.cpu().numpy()
    target = target.cpu().numpy()
    mae_batch = np.mean(np.abs(predictions-target) ,  axis=(0,1)).sum()
    return mae_batch

def mssim(predictions, target):
    predictions, target = predictions.cpu().numpy(), target.cpu().numpy()
    total_ssim = 0
    for a in range(0,target.shape[0]):
        for b in range(0,target.shape[1]):
            total_ssim += ssim(target[a,b,0,], predictions[a,b,0,])
    return total_ssim/ (target.shape[0]*target.shape[1])

def bce(predictions, target):
    eps = 1e-4
    predictions, target = predictions.cpu().numpy(), target.cpu().numpy()
    predictions[predictions < eps] = eps
    predictions[predictions > 1 - eps] = 1 -eps
    target[target < eps] = eps
    target[target > 1 - eps] = 1 -eps

    batch_size, target_length = predictions.shape[0:2]
    cross_entropy = -target*np.log(predictions) - (1-target) * np.log(1-predictions)
    cross_entropy = cross_entropy.sum()
    cross_entropy = cross_entropy / (batch_size*target_length)
    return cross_entropy

def fvd_session(predictions, target):
    # 40, 10, 1, h, w

    # with tf.Graph().as_default():
    NUMBER_OF_VIDEOS, VIDEO_LENGTH, FRAME_WIDTH, FRAME_HEIGHT, C = predictions.shape
    # with values in 0-255
    assert C == 3

    result = fvd.calculate_fvd(
        fvd.create_id3_embedding(fvd.preprocess(predictions,
                                                (224, 224))),
        fvd.create_id3_embedding(fvd.preprocess(target,
                                                (224, 224))))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        print("FVD is: %.2f." % sess.run(result))
        return result


def mfvd(predictions, target):
    predictions,  target = [tf.convert_to_tensor((item*255).permute(0,1,3,4,2).repeat_interleave(3, dim=-1).cpu().numpy()) for item in [predictions, target]]
    tf.app.run(fvd_session(predictions, target))

loss_fn = lpips.LPIPS(net='alex')

def mlpips(predictions, target):
    # Variables im0, im1 is a PyTorch Tensor/Variable with shape Nx3xHxW
    # (N patches of size HxW, RGB images scaled in [-1,+1]). This returns d, a length N Tensor/Variable.
    bs, T, C, H, W = predictions.shape
    predictions,  target = [(item*2 - 1).reshape(bs*T, C, H, W).repeat_interleave(3, dim=1) for item in [predictions, target]]
    if not hasattr(loss_fn, 'device'):
        loss_fn.to(predictions.device)# TODO: fer global?
    d = loss_fn.forward(predictions, target).mean().cpu().numpy()
    return d