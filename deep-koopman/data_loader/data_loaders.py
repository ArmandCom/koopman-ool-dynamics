from torchvision import datasets, transforms
from base import BaseDataLoader
from data_loader.ground_truth import named_data
from data_loader.dlib_dataset import DisentanglementDataset

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
    def __init__(self, dataset_name, seq_length, seq_stride, data_dir, batch_size, shuffle=True, training_split=0.9, validation_split=0.0, num_workers=1, training=True):

        self.data_dir = data_dir
        self.name = dataset_name
        self.ground_truth_class = named_data.get_named_ground_truth_data(self.name, self.data_dir)
        self.dataset = DisentanglementDataset(self.ground_truth_class, seq_length, seq_stride,
                                              train=training, train_split=training_split)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)