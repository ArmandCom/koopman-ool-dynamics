import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler, BatchSampler
from torch.utils.data import Sampler
from PIL import Image


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, n_objects_to_repeat, validation_split, dataset_reduction, num_workers, collate_fn=default_collate, training=True):
        self.validation_split = validation_split
        self.shuffle = shuffle

        # self.batch_idx = 0

        # self.n_samples = len(dataset.images)
        self.n_samples = dataset.length

        if training:
            self.sampler, self.valid_sampler = self._split_sampler(self.validation_split, dataset_reduction, n_objects_to_repeat)
        else:
            # _, self.sampler = self._split_sampler(self.validation_split, dataset_reduction, n_objects_to_repeat)
            self.sampler = None #Sampler(np.arange(test_size))

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split, dataset_reduction=0, n_objects=1):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)
        idx_full = idx_full[:int((1-dataset_reduction) * self.n_samples)]

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split * (1 - dataset_reduction))

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        valid_idx = np.concatenate([valid_idx]*n_objects, axis=0)
        train_idx = np.concatenate([train_idx]*n_objects, axis=0)

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        # WeightedRandomSampler doesn't work as expected.
        # valid_weights = np.zeros(self.n_samples)
        # train_weights = np.zeros(self.n_samples)
        # valid_weights[valid_idx], train_weights[train_idx] = 1, 1
        # train_sampler = WeightedRandomSampler(train_weights, num_samples=n_objects)
        # valid_sampler = WeightedRandomSampler(valid_weights, num_samples=n_objects)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)
