import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class DisentanglementDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, ground_truth_class, seq_length, stride, train=True, train_split=0.8, transform=None):

        self.data_class = ground_truth_class
        self.transform = transform
        self.seq_len = seq_length
        self.stride = stride
        self.train = train
        self.train_split = train_split
        self._calculate_factor_space_length()

        #Split between train and test sets
        idx_full = np.arange(self.length)
        np.random.seed(0)
        np.random.shuffle(idx_full)

        len_test = int((1-train_split) * self.length)
        test_idx = idx_full[0:len_test]
        train_idx = np.delete(idx_full, np.arange(0, len_test))

        if train:
            self.split = train_idx
        else:
            self.split = test_idx

        # For quick testing of the main loading functions
        self.random_state = np.random.RandomState()
        # index = self.random_state.randint(self.length)
        # sample = self._generate_training_sample(self.data_class, index=index)

    def __len__(self):
        return len(self.split)

    def __getitem__(self, idx):

        split_index = self.split[idx]
        sample = self._generate_training_sample(self.data_class, index=split_index)
        sample = sample[:, None, :, :, 0]
        if self.transform:
            sample = self.transform(sample)

        return sample

    def _calculate_factor_space_length(self):
        self.usable_factor_lengths = []
        self.swipe_factor_bool = np.zeros(len(self.data_class.factors_num_values))
        self.num_varying_factors = 0
        for idx, item in enumerate(self.data_class.factors_num_values):
            if item > self.seq_len * self.stride:
                self.usable_factor_lengths.append((item - self.seq_len) // self.stride)
                self.swipe_factor_bool[idx] = 1
                self.num_varying_factors += 1
            else:
                self.usable_factor_lengths.append(item)
        if self.num_varying_factors == 0:
            raise Exception("No varying factors in dataset!!!")
        self.len_factor_space = np.prod(self.usable_factor_lengths)
        self.len_activation_and_direction = 3 ** self.num_varying_factors - 1
        self.length = self.len_factor_space * self.len_activation_and_direction

    def _to_ternary_list(self, n):
        # Note: num has to be added + 1 as there must be at least one factor of variation
        if n == 0:
            return '0 is not valid'
        nums = []
        while n:
            n, r = divmod(n, 3)
            nums.append(r)
        if len(nums) < self.num_varying_factors:
            [nums.append(0) for i in range(self.num_varying_factors-len(nums))]
        nums.reverse()
        assert len(nums) == self.num_varying_factors
        return nums

    def _get_initial_factors_and_direction(self, index):
        dir_and_act = index % self.len_activation_and_direction
        dir_and_act_tern = self._to_ternary_list(dir_and_act + 1)

        factors = index // self.len_activation_and_direction
        factor_values = []
        for factor in self.usable_factor_lengths:
            factor_values.append(factors % factor)
            factors = factors // factor
        if len(factor_values) < len(self.data_class.factors_num_values):
            [factor_values.append(0) for i in range(len(self.data_class.factors_num_values) - len(factor_values))]
        assert len(factor_values) == len(self.data_class.factors_num_values)

        return factor_values, dir_and_act_tern

    def _get_factors(self, ini_factors, directions):
        all_factors = np.zeros(
            shape=(self.seq_len, len(ini_factors)), dtype=np.int64)
        count = 0
        for fac_id, factor in enumerate(ini_factors):

            if self.swipe_factor_bool[fac_id]:
                if directions[count] == 0:
                    all_factors[:, fac_id] = factor
                else:
                    sweep = list(range(factor, factor + self.stride*self.seq_len, self.stride))
                    if directions[count] == 2:
                        sweep.reverse()
                    all_factors[:, fac_id] = sweep
                count += 1
            else:
                all_factors[:, fac_id] = factor
        return all_factors

    def _generate_training_sample(self, ground_truth_data, index):
        """Sample a single training sample based on a mini-batch of ground-truth data.

        Args:
        ground_truth_data: GroundTruthData to be sampled from.
        index: dataset index

        Returns:
        sample: Video sample

        """
        ini_factors, directions = self._get_initial_factors_and_direction(index)
        factors = self._get_factors(ini_factors, directions)

        sample = ground_truth_data.sample_observations_from_factors(
          factors, self.random_state)

        # Note: Save sample video
        # _save_example_observation(sample, self.seq_len)

        return sample


def _save_example_observation(sequence, length):
    seq = (sequence.reshape(length * 64, 64) * 255)
    im = Image.fromarray(seq)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    im.save("example_observation.jpg")