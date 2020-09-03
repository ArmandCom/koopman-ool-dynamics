import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class DisentanglementDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, ground_truth_class, transform=None):

        self.data_class = ground_truth_class
        self.transform = transform
        self.seq_len = 10
        self.stride = 1
        self._calculate_factor_space_length()


        self.random_state = np.random.RandomState()
        index = self.random_state.randint(self.length)

        sample = self._generate_training_sample(self.data_class,
                              seq_len=self.seq_len, index=index, random_state=self.random_state)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        sample = self._generate_training_sample(self.data_class,
                              seq_len=self.seq_len, index=idx, random_state=self.random_state)

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
                    sweep = list(range(factor, factor + self.seq_len, self.stride))
                    if directions[count] == 2:
                        sweep.reverse()
                    all_factors[:, fac_id] = sweep
                count+=1
            else:
                all_factors[:, fac_id] = factor
        return all_factors

    def _generate_training_sample(self, ground_truth_data,
                                  seq_len, index, random_state):
      """Sample a single training sample based on a mini-batch of ground-truth data.

      Args:
        ground_truth_data: GroundTruthData to be sampled from.
        representation_function: Function that takes observation as input and
          outputs a representation.
        batch_size: Number of points to be used to compute the training_sample
        random_state: Numpy random state used for randomness.

      Returns:
        index: Index of coordinate to be used.
        feature_vector: Feature vector of training sample.
      """
      ini_factors, directions = self._get_initial_factors_and_direction(index)
      factors = self._get_factors(ini_factors, directions)

      sample = ground_truth_data.sample_observations_from_factors(
          factors, self.random_state)

      # _save_example_observation(observation1, seq_len)

      return sample


def _save_example_observation(sequence, length):
    seq = (sequence.reshape(length * 64, 64) * 255)
    im = Image.fromarray(seq)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    im.save("example_observation.jpg")