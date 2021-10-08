import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from torch.utils.data import TensorDataset, DataLoader
import hydra
from omegaconf import DictConfig
import numpy as np
import os
# import scipy.misc
from PIL import Image as im
from utils.util import plot_matrix

from utils import overlap_objects_from_batch

def remove_eig_under_t(A, t=0.8):
    e, V = np.linalg.eig(A.cpu().detach().numpy())
    s = np.abs(e)
    indices = s < t

    mask = np.ones_like(e)
    mask[6:] = 0
    #     plot_name = remove_plot_name[ind] +'_eig'
    # else:
    #     mask = np.zeros_like(e)
    #     mask[keep_indices[ind]] = 1
    #     plot_name = keep_plot_name[ind] + '_eig'
    e = e*mask

    # plot_complex_number(e[mask==1], image_save_path + plot_name+'_eigplot.png')
    A_modified = np.matmul((np.matmul(V, np.diag(e))),np.linalg.inv(V))
    print('Number of eigenvalues: ', mask.sum())
    # exit()
    return A_modified, indices, e

# TODO: Full test function has to be modified according to the new changes.
@hydra.main(config_path="conf/config.yaml")
def main(cfg_dict : DictConfig):
    generate = False
    load_gen = True
    save = True
    # remove_eigs = True
    remove_eigs = False

    config = ConfigParser(cfg_dict)
    T_rec, T_pred = config['n_timesteps'], config['seq_length'] - config['n_timesteps']
    logger = config.get_logger('test')

    gt = True
    # gt = True
    model_name = 'ddpae-iccv'
    # model_name = 'DRNET'
    # model_name = 'scalor'
    # model_name = 'sqair'
    s_directory = os.path.join(config['data_loader']['args']['data_dir'], 'test_data')
    res_directory = os.path.join(config['data_loader']['args']['data_dir'], 'res_data')
    load_gen_directory = os.path.join(config['data_loader']['args']['data_dir'],
                                      'results')
    # # TODO: Testing features
    # load_gen_directory = os.path.join(config['data_loader']['args']['data_dir'], 'test_data')

    if not os.path.exists(s_directory):
        os.makedirs(s_directory)
    if not os.path.exists(res_directory):
        os.makedirs(res_directory)
    dataset_dir = os.path.join(s_directory, config['data_loader']['args']['dataset_case']+
                               '_Len-'+str(config['seq_length'])+'_Nts-'+str(config['n_timesteps'])+'.npy')
    results_dir = os.path.join(res_directory, config['data_loader']['args']['dataset_case']+
                               '_Len-'+str(config['seq_length'])+'_Nts-'+str(config['n_timesteps'])+'.npz')
    all_data = []
    if not os.path.exists(dataset_dir) and generate:
        config['data_loader']['args']['shuffle'] = False
        config['data_loader']['args']['training'] = False
        config['data_loader']['args']['validation_split'] = 0.0
        data_loader = config.init_obj('data_loader', module_data)

        for i, data in enumerate(tqdm(data_loader)):
            all_data.append(data)
        all_data = torch.cat(all_data, dim=0).numpy()
        print(all_data.shape)
        np.save(dataset_dir, all_data)
        print(config['data_loader']['args']['dataset_case']+ ' data generated in: '+s_directory)
        exit()
    if os.path.exists(dataset_dir):
        print('LOADING EXISTING DATA FROM: ' + dataset_dir)
        inps = torch.from_numpy(np.load(dataset_dir))
        if os.path.exists(load_gen_directory) and load_gen:
            if model_name == 'ddpae-iccv':
                outs = torch.from_numpy(
                    np.load(os.path.join(
                        load_gen_directory,
                        model_name +'--'+config['data_loader']['args']['dataset_case']+
                        '_Len-'+str(config['seq_length'])+'_Nts-'+str(config['n_timesteps'])+'.npy')))

            else:
                with np.load(os.path.join(
                        load_gen_directory,
                        model_name +'_'+config['data_loader']['args']['dataset_case']+'.npz')) as outputs:
                    if model_name == 'scalor':
                        outs = torch.from_numpy(outputs["pred"]).permute(0,1,3,2).unsqueeze(2)
                    elif model_name == 'DRNET':
                        outs = torch.from_numpy(outputs["pred"]).unsqueeze(2).float()
                    else:
                        outs = torch.from_numpy(outputs["pred"]).unsqueeze(2)
                    print('Inps and Outs shapes', inps.shape, outs.shape)
            loaded_dataset = TensorDataset(inps, outs)
        else:
            loaded_dataset = TensorDataset(inps)
        data_loader = DataLoader(loaded_dataset, batch_size=40, shuffle=False, sampler=None,
                            batch_sampler=None, num_workers=2, collate_fn=None,
                            pin_memory=False)
    else:
        print('te has liao si te metes aqui')
        exit()
        config['data_loader']['args']['shuffle'] = False
        config['data_loader']['args']['training'] = False
        config['data_loader']['args']['validation_split'] = 0.0
        data_loader = config.init_obj('data_loader', module_data)
    # build model architecture
    if not load_gen:
        model = config.init_obj('arch', module_arch)
    # logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in ["mse", "mae", "bce", "mssim", "mlpips"]]

    if not load_gen:
        logger.info('Loading checkpoint: {} ...'.format(config['resume']))
        checkpoint = torch.load(config['resume'])
        state_dict = checkpoint['state_dict']
        if config['n_gpu'] > 1:
            model = torch.nn.DataParallel(model)
        model.load_state_dict(state_dict)

        # prepare model for testing
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()

        if remove_eigs:
            A_modified, indices, e = remove_eig_under_t(
                model.koopman.dynamics.dynamics.weight.data, t=0.7)
            A_modified = torch.from_numpy(A_modified.real).to(device)
            model.koopman.dynamics.dynamics.weight.data = A_modified

    total_loss = 0.0


    total_metrics = [torch.zeros(len(metric_fns)), torch.zeros(len(metric_fns))]

    # TODO: Here we can change the model's K, and crop the eigenvalues under certain module threshold.
    # Si la nova prediccio es mes llarga, evaluem nomes la nova:
    # T_pred = 8
    all_pred, all_rec = [], []
    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader)):
            if isinstance(data, list) and len(data) == 2:
                target = data[0]
                output = data[1]
                batch_size = target.shape[0]
                # total_loss += loss.item() * batch_size
                pred = output[:, -T_pred:], target[:, -T_pred:]
                rec = output[:, :T_rec], target[:, :T_rec]

                assert T_rec + T_pred == target.shape[1]
                assert target.shape == output.shape
            else:
                if isinstance(data, list) and len(data) == 1:
                    data = data[0]
                # if config["data_loader"]["type"] == "MovingMNISTLoader":
                #     data = overlap_objects_from_batch(data,config['n_objects'])
                target = data # Is data a variable?
                data, target = data.to(device), target.to(device)

                output = model(data, epoch_iter=[-1], test=True)
                # computing loss, metrics on test set
                # loss, loss_particles = loss_fn(output, target,
                #                                epoch_iter=[-1],
                #                                case=config["data_loader"]["args"]["dataset_case"])
                batch_size = data.shape[0]
                # total_loss += loss.item() * batch_size

                pred = output["pred_roll"][:, -T_pred:] , target[:, -T_pred:] #* 0.85
                rec = output["rec_ori"][:, :T_rec] * 0.85, target[:, :T_rec]

                assert T_rec + T_pred == target.shape[1]

            if config['data_loader']['args']['dataset_case'] == 'circles_crop':
                rec_cr, pred_cr = [crop_top_left_keepdim(vid[0], 35) for vid in [rec, pred]]
                rec, pred = (rec_cr, target[:, :T_rec]), (pred_cr, target[:, -T_pred:])

            # Save image sample
            if i==0:
                if gt:
                    idx_gt = 1
                else:
                    idx_gt = 0
                # 11 fail to reconstruct.
                idx = 21
                # print(rec.shape, pred.shape)
                # print_u = output["u"].reshape(40, 2, -1, 4)[idx,:,-torch.cat(pred, dim=-2).shape[1]:]\
                #     .cpu()
                # print_u = print_u.abs()*255
                # print_im = torch.cat(pred, dim=-2).permute(0,2,3,1,4)[idx,0,:,:]
                print_im = pred[idx_gt].permute(0,2,3,1,4)[idx,0]
                np.save("/home/acomasma/ool-dynamics/dk/image_sample.npy", print_im.cpu().numpy())
                image = im.fromarray(print_im.reshape(print_im.shape[-3], -1).cpu().numpy()*255)
                image = image.convert('RGB')
                image.save("/home/acomasma/ool-dynamics/dk/image_sample.png")

                # u_plot_o1 = im.fromarray(plot_matrix(print_u[0]).permute(1,0).numpy()).convert('RGB')
                # u_plot_o1.save("/home/acomasma/ool-dynamics/dk/input_sample_o1.png")
                #
                # u_plot_o2 = im.fromarray(plot_matrix(print_u[1]).permute(1,0).numpy()).convert('RGB')
                # u_plot_o2.save("/home/acomasma/ool-dynamics/dk/input_sample_o2.png")
                # exit()
                image = im.fromarray(rec[idx_gt].permute(0,2,3,1,4)[idx,0].reshape(64, -1).cpu().numpy()*255)
                image = image.convert('RGB')
                image.save("/home/acomasma/ool-dynamics/dk/image_sample_rec.png")
                exit()

            all_pred.append(pred[0])
            all_rec.append(rec[0])

            for j, (out, tar) in enumerate([rec, pred]):
                for i, metric in enumerate(metric_fns):
                    # TODO: dataset case in metrics
                    total_metrics[j][i] += metric(out, tar) * batch_size

    n_samples = len(data_loader.sampler)
    print('n_samples', n_samples)
    # log = {'loss': total_loss / n_samples}
    log = {}

    print('Timesteps Rec and pred: ' , T_rec, T_pred)
    for j, name in enumerate(['rec', 'pred']):
        log.update({
            met.__name__: total_metrics[j][i].item() / n_samples for i, met in enumerate(metric_fns)
        })
        print(name)
        logger.info(log)
    # if save:
    #     dict_out = {'pred': torch.cat(all_pred, dim=0), 'rec': torch.cat(all_rec, dim=0)}
    #     np.savez(results_dir, dict_out)
    #     print_im = torch.cat(rec, dim=-2).permute(0,2,3,1,4)[0,0]
    #     image = im.fromarray(print_im.reshape(print_im.shape[-3], -1).cpu().numpy()*255)
    #     image = image.convert('RGB')
    #     image.save("/home/acomasma/ool-dynamics/dk/image_sample.png")

def crop_top_left_keepdim(img, cropx):
    y, x = img.shape[-2:]
    mask = torch.ones_like(img)
    # mask[..., :, :(x - cropx)] = 0
    mask[..., :(y - cropx), :] = 0
    return img * mask

if __name__ == '__main__':
    main()

