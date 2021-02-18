import numpy as np
import torch
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_tensor
from base import BaseTrainer
from utils import inf_loop, plot_representation, plot_matrix, MetricTracker, overlap_objects_from_batch


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler

        if self.config['log_step'] is not None:
            self.log_step = self.config['log_step']
        else:
            self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()

        for batch_idx, data in enumerate(self.data_loader):
            if self.config["data_loader"]["type"] == "MovingMNISTLoader":
                data = overlap_objects_from_batch(data, self.config['n_objects'])
            target = data # Is data a variable?
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data, epoch)
            loss, loss_particles = self.criterion(output, target,
                                                  epoch_iter=(epoch, (epoch-1)*len(self.data_loader)+batch_idx), lambd=self.config["trainer"]["lambd"])
            loss = loss.mean()

            # Note: from space implementation
            # optimizer_fg.zero_grad()
            # optimizer_bg.zero_grad()
            # loss.backward()
            # if cfg.train.clip_norm:
            #     clip_grad_norm_(model.parameters(), cfg.train.clip_norm)

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1000)
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                loss_particles_str = " ".join([key + ': {:.2f}, '.format(loss_particles[key].item()) for key in loss_particles])

                self.logger.debug('Train Epoch: {} {} '.format(epoch, self._progress(batch_idx)) + loss_particles_str + 'Loss: {:.6f}'.format(
                    loss.item()))

                self._show(data, output)

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            # loss = 0 # Note: Only when training is commented
            self.lr_scheduler.step(loss)
            # self.lr_scheduler.step() #Note: If it doesn't require argument.
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'])
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, data in enumerate(self.valid_data_loader):
                data = overlap_objects_from_batch(data, self.config['n_objects'])
                target = data  # Is data a variable?
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data, epoch=epoch)
                loss, loss_particles = self.criterion(output, target,
                                                      epoch_iter=(epoch, (epoch + 1)*batch_idx),
                                                      lambd=self.config["trainer"]["lambd"])

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))

                self._show(data, output, train=False)

        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _show(self, data, output, train=True):

        self.writer.add_image('a-input', make_grid(data[0].cpu(), nrow=data.shape[1], normalize=True))
        self.writer.add_image('b-output_rec', make_grid(output["rec"][0].cpu(), nrow=output["rec"].shape[1], normalize=True))
        self.writer.add_image('c-output_pred', make_grid(output["pred"][0].cpu(), nrow=output["pred"].shape[1], normalize=True))

        if output["selector"] is not None:
            S_plot = plot_matrix(output["selector"])
            self.writer.add_image('ba-S', make_grid(S_plot, nrow=1, normalize=False))
        if output["obs_rec_pred"] is not None:
            g_plot = plot_representation        (output["obs_rec_pred"][:,:output["rec"].shape[1]].cpu())
            g_plot_pred = plot_representation   (output["obs_rec_pred"][:,output["rec"].shape[1]:].cpu())
            self.writer.add_image('d-g_repr_rec', make_grid(to_tensor(g_plot), nrow=1, normalize=False))
            self.writer.add_image('e-g_repr_pred', make_grid(to_tensor(g_plot_pred), nrow=1, normalize=False))
        if output["obs_rec_pred_rev"] is not None:
            g_plot = plot_representation        (output["obs_rec_pred_rev"].cpu())
            self.writer.add_image('ea-g_repr_pred_rev', make_grid(to_tensor(g_plot), nrow=1, normalize=False))
        if output["A"] is not None:
            A_plot = plot_matrix(output["A"])
            self.writer.add_image('f-A', make_grid(A_plot, nrow=1, normalize=False))
        if output["Ainv"] is not None:
            A_plot = plot_matrix(output["Ainv"])
            self.writer.add_image('fa-A-inv', make_grid(A_plot, nrow=1, normalize=False))
            AA_plot = plot_matrix(torch.mm(output["A"],output["Ainv"]))
            self.writer.add_image('fb-AA', make_grid(AA_plot, nrow=1, normalize=False))
        # else:
        #     AA_plot = plot_matrix(torch.mm(output["A"],torch.pinverse(output["A"])))
        #     self.writer.add_image('fb-AA', make_grid(AA_plot, nrow=1, normalize=False))
        if output["B"] is not None:
            B_plot = plot_matrix(output["B"])
            self.writer.add_image('g-B', make_grid(B_plot, nrow=1, normalize=False))
        if output["u"] is not None:
            u_plot = plot_representation(output["u"][:, :output["u"].shape[1]].cpu())
            self.writer.add_image('eb-gu', make_grid(to_tensor(u_plot), nrow=1, normalize=False))

        # if output[10] is not None: # TODO: Ara el torno a posar
        #     # print(output[10][0].max(), output[-1][0].min())
        #     shape = output[10][0].shape
        #     self.writer.add_image('objects', make_grid(output[10][0].permute(1, 2, 0, 3, 4).reshape(*shape[1:-2], -1, shape[-1]).cpu(), nrow=output[0].shape[1], normalize=True))

        # output["rec"] = torch.clamp(torch.sum(out_rec.reshape(bs, self.n_objects, -1, *out_rec.size()[1:]), dim=1), min=0, max=1)
        # output["pred"] = torch.clamp(torch.sum(out_pred.reshape(bs, self.n_objects, -1, *out_rec.size()[1:]), dim=1), min=0, max=1)
        # output["obs_rec_pred"] = returned_g.reshape(-1, returned_g.size()[-1:])
        # output["gauss_post"] = returned_post
        # output["A"] = A
        # output["B"] = B
        # output["u"] = u.reshape(bs * self.n_objects, -1, u.shape[-1])
        # output["bern_logit"] = u_logit.reshape(bs, self.n_objects, -1, u_logit.shape[-1]) # Input distribution