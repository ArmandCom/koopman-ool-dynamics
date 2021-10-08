import json
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import io
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.distributions import RelaxedBernoulli
from torch.distributions.utils import broadcast_all
from torch.autograd import Function
import torch.nn as nn

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def print_var_shape(var, var_name, q=False):
    if isinstance(var, list):
        for v in var:
            if v is None:
                print(var_name + ': None')
            else:
                print(var_name + ": {}".format(v.shape))
    else:
        if var is None:
            print('None')
        else:
            print(var_name + ": {}".format(var.shape))
    if q:
        exit()

def plot_representation(repr, instance=0):  # TODO: does this function belong here? Maybe log
    assert len(repr.shape) == 3
    repr = repr.detach().numpy()
    [plt.plot(range(repr.shape[1]), repr[instance, :, f_idx]) for f_idx in range(repr.shape[-1])]
    plt.xlabel('t')
    plt.ylabel('Mag')

    plt.title("Observables evolution")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    img = Image.open(buf)

    return img

def plot_matrix(M, instance=0):
    if len(M.shape)==4:
        M = block_diagonal(M)
    if len(M.shape)==2:
        M = M.unsqueeze(0)
        instance=0
    expand = 20
    side = M.shape[-1]
    M = M[instance, :, None, :, None].repeat(1, expand, 1, expand)
    M = M.reshape(-1, side * expand)

    # M = torch.abs(M)
    # M = M / M.max()
    return M

def block_diagonal(A):
    # TODO: make sure we are showing the propper dimensions.
    #   It could be that we are using different dimensions for mu and logvar as expected here

    bs, num_blocks, first_dim_size, second_dim_size = A.shape
    # num_blocks = num_blocks//2
    A = A[:,:num_blocks]
    block_A = torch.zeros(bs, first_dim_size*num_blocks, second_dim_size*num_blocks)

    for i in range(num_blocks):
        block_A[:, i*first_dim_size:(i*first_dim_size)+first_dim_size,
        i*second_dim_size:(i*second_dim_size)+second_dim_size] \
            = A[:,i]
    return block_A

def overlap_objects_from_batch(data, n_objects):
    batch_size = data.shape[0]
    real_batch_size = batch_size//n_objects
    data_new = data.reshape(real_batch_size, n_objects, *data.shape[1:])
    data_new = torch.sum(data_new, dim=1)
    return torch.clamp(data_new, min=0, max=1)

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()
        
    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]
    
    def result(self):
        return dict(self._data.average)


class NumericalRelaxedBernoulli(RelaxedBernoulli):
    """
    This is a bit weird. In essence it is just RelaxedBernoulli with logit as input.
    """

    def rsample(self, sample_shape=torch.Size()):
        return self.base_dist.rsample(sample_shape)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        logits, value = broadcast_all(self.logits, value)
        diff = logits - value.mul(self.temperature)

        out = self.temperature.log() + diff - 2 * diff.exp().log1p()

        return out

def linear_annealing(device, step, start_step, end_step, start_value, end_value):
    """
    Linear annealing

    :param x: original value. Only for getting device
    :param step: current global step
    :param start_step: when to start changing value
    :param end_step: when to stop changing value
    :param start_value: initial value
    :param end_value: final value
    :return:
    """
    if step <= start_step:
        x = torch.tensor(start_value, device=device)
    elif start_step < step < end_step:
        slope = (end_value - start_value) / (end_step - start_step)
        x = torch.tensor(start_value + slope * (step - start_step), device=device)
    else:
        x = torch.tensor(end_value, device=device)

    return x
