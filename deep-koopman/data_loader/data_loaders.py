from torchvision import datasets, transforms
import torch
from base import BaseDataLoader
from data_loader.ground_truth import named_data
from data_loader.dlib_dataset import DisentanglementDataset
from data_loader.moving_mnist_dataset import MovingMNISTDataset
import numpy as np

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        # TODO: this data directory is under outputs for Hydra. Set absolute in config / find out how to ignore working dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class DlibLoader(BaseDataLoader):
    """
    disentanglement_lib dataset loading using BaseDataLoader
    """
    def __init__(self, dataset_name, seq_length, seq_stride, n_objects, data_dir, batch_size, shuffle=True, training_split=0.9, validation_split=0.0, dataset_reduction=0, num_workers=1, training=True):


        total_batch_size = batch_size * n_objects #This is a patchy way of doing it
        self.data_dir = data_dir
        self.name = dataset_name
        self.ground_truth_class = named_data.get_named_ground_truth_data(self.name, self.data_dir)

        self.dataset = DisentanglementDataset(self.ground_truth_class, seq_length, seq_stride,
                                              train=training, train_split=training_split)

        super().__init__(self.dataset, total_batch_size, shuffle, n_objects, validation_split, dataset_reduction, num_workers)

class MovingMNISTLoader(BaseDataLoader):
    """
    disentanglement_lib dataset loading using BaseDataLoader
    """
    def __init__(self, dataset_name, seq_length, seq_stride, n_objects, data_dir, batch_size, shuffle=True, training_split=0.9, validation_split=0.0, dataset_reduction=0, num_workers=1, training=True):


        total_batch_size = batch_size * n_objects #This is a patchy way of doing it
        self.data_dir = data_dir
        self.name = dataset_name
        transform = transforms.Compose([ToTensor()])
        self.dataset = MovingMNISTDataset(data_dir, training, seq_length,
                                   0, 1, training_split, transform) #TODO: we set num_obj to 1 for all cases, to merge them in the patchy way

        super().__init__(self.dataset, total_batch_size, shuffle, n_objects, validation_split, dataset_reduction, num_workers)

class ToTensor(object):
    """Converts a numpy.ndarray (... x H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (... x C x H x W) in the range [0.0, 1.0].
    """
    def __init__(self, scale=True):
        self.scale = scale

    def __call__(self, arr):
        if isinstance(arr, np.ndarray):
            video = torch.from_numpy(np.rollaxis(arr, axis=-1, start=-3))

            if self.scale:
                return video.float().div(255)
            else:
                return video.float()
        else:
            raise NotImplementedError