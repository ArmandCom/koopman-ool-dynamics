from torchvision import datasets, transforms
import torch
from base import BaseDataLoader
from data_loader.ground_truth import named_data
from data_loader.dlib_dataset import DisentanglementDataset
from data_loader.moving_mnist_dataset import MovingMNISTDataset
from data_loader.bouncing_balls_dataset import BouncingBallsDataset
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

class BouncingBallsLoader(BaseDataLoader):
    """
    disentanglement_lib dataset loading using BaseDataLoader
    """
    def __init__(self, dataset_name, seq_length, seq_stride, n_objects, data_dir, batch_size, image_size, shuffle=True, training_split=0.9, validation_split=0.0, dataset_reduction=0, num_workers=1, training=True):


        total_batch_size = batch_size
        self.data_dir = data_dir
        self.name = dataset_name
        transform = transforms.Compose([ToTensor()]) #Scale()
        self.dataset = BouncingBallsDataset(data_dir, training, seq_length,
                                          0, n_objects, image_size, training_split, transform)
        super().__init__(self.dataset, total_batch_size, shuffle, 1, validation_split, dataset_reduction, num_workers)

# elif opt.dset_name == 'bouncing_balls':
# transform = transforms.Compose([vtransforms.Scale(opt.image_size),
#                                 vtransforms.ToTensor()])
# dset = BouncingBalls(opt.dset_path, opt.is_train, opt.n_frames_input,
#                      opt.n_frames_output, opt.image_size[0], transform)

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

class Scale(object):
    """Rescale the input numpy.ndarray to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``bilinear``
    """

    def __init__(self, size, interpolation='bilinear'):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, video):
        """
        Args:
            video (numpy.ndarray): Video to be scaled.
        Returns:
            numpy.ndarray: Rescaled video.
        """
        if isinstance(self.size, int):
            w, h = video.shape[-2], video.shape[-3]
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return video
            if w < h:
                ow = self.size
                oh = int(self.size*h/w)
                return resize(video, (ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size*w/h)
                return resize(video, (ow, oh), self.interpolation)
        else:
            return resize(video, self.size, self.interpolation)

def resize(video, size, interpolation):
    if interpolation == 'bilinear':
        inter = cv2.INTER_LINEAR
    elif interpolation == 'nearest':
        inter = cv2.INTER_NEAREST
    else:
        raise NotImplementedError

    shape = video.shape[:-3]
    video = video.reshape((-1, *video.shape[-3:]))
    resized_video = np.zeros((video.shape[0], size[1], size[0], video.shape[-1]))
    for i in range(video.shape[0]):
        img = cv2.resize(video[i], size, inter)
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
        resized_video[i] = img
    return resized_video.reshape((*shape, size[1], size[0], video.shape[-1]))
