import gzip
import math
import numpy as np
import os
from PIL import Image
import random
import torch
import torch.utils.data as data

def load_mnist(root):
    # Load MNIST dataset for generating training data.
    path = os.path.join(root, 'moving_mnist/train-images-idx3-ubyte.gz')
    with gzip.open(path, 'rb') as f:
        mnist = np.frombuffer(f.read(), np.uint8, offset=16)
        mnist = mnist.reshape(-1, 28, 28)
    return mnist

def load_fixed_set(root, is_train):
    # Load the fixed dataset
    filename = 'mnist_test_seq.npy'
    path = os.path.join(root, filename)
    dataset = np.load(path)
    dataset = dataset[..., np.newaxis]
    return dataset

class MovingMNISTDataset(data.Dataset):
    def __init__(self, root, train, n_frames_input, n_frames_output, num_objects,
                 train_split=0.8, transform=None):
        '''
        param num_objects: a list of number of possible objects.
        '''
        super(MovingMNISTDataset, self).__init__()

        self.motion_type = 'constant_vel'
        # self.motion_type = 'circles'

        self.dataset = None
        # if is_train:
        #     self.mnist = load_mnist(root)
        # else:
        #     if num_objects[0] != 2:
        #         self.mnist = load_mnist(root)
        #     else:
        #         self.dataset = load_fixed_set(root, False)

        self.length = 4e4
        idx_full = np.arange(self.length)
        np.random.seed(0)
        np.random.shuffle(idx_full)

        # len_test = int((1-train_split) * self.length)
        # test_idx = idx_full[0:len_test]
        # train_idx = np.delete(idx_full, np.arange(0, len_test))
        # if train:
        #     self.split = train_idx
        # else:
        #     self.split = test_idx

        self.split = idx_full

        self.mnist = load_mnist(root)

        self.is_train = train
        self.num_objects = num_objects
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.n_frames_total = self.n_frames_input + self.n_frames_output
        self.transform = transform
        # For generating data
        self.image_size_ = 128
        self.digit_size_ = 28
        self.step_length_ = 0.2 # was 0.25

    def get_random_trajectory(self, seq_length, motion_type='constant_vel'):
        ''' Generate a random sequence of a MNIST digit '''
        canvas_size = self.image_size_ - self.digit_size_
        x = random.random()
        y = random.random()
        theta = random.random() * 2 * np.pi
        v_y = np.sin(theta)
        v_x = np.cos(theta)
        # v_x = 0

        start_y = np.zeros(seq_length)
        start_x = np.zeros(seq_length)

        if motion_type == 'circles':
            y = y * 0.6 + 0.2
            x = x * 0.6 + 0.2

            R = random.random()*0.4 + 0.2
            if x + R >= 1.0:
                R = 1 - random.random()*0.1 - x
            if x - R <= 0:
                R = x - random.random()*0.1
            if y + R >= 1.0:
                R = 1 - random.random()*0.1 - y
            if y - R <= 0:
                R = y - random.random()*0.1

            factor = 2
            t = np.linspace(0, seq_length * self.step_length_ * factor, seq_length)

            if random.random() < 0.5:
                t = np.flip(t, axis=0)

            start_x, start_y = R*np.cos(t) + x, R*np.sin(t) + y

        if motion_type == 'constant_vel':
            elc = 1.2 # was 1.1 without inversions
            for i in range(seq_length):
                # Take a step along velocity.
                y += v_y * self.step_length_
                x += v_x * self.step_length_

                # Bounce off edges.
                if x <= 0:
                    x = 0
                    v_x = -v_x * elc
                if x >= 1.0:
                    x = 1.0
                    v_x = -v_x * (1/elc)
                if y <= 0:
                    y = 0
                    v_y = -v_y * elc
                if y >= 1.0:
                    y = 1.0
                    v_y = -v_y * (1/elc)
                start_y[i] = y
                start_x[i] = x

        # Scale to the size of the canvas.
        start_y = (canvas_size * start_y).astype(np.int32)
        start_x = (canvas_size * start_x).astype(np.int32)
        return start_y, start_x

    def generate_moving_mnist(self, num_digits=2):
        '''
        Get random trajectories for the digits and generate a video.
        '''
        data = np.zeros((self.n_frames_total, self.image_size_, self.image_size_), dtype=np.float32)
        for n in range(num_digits):
            # Trajectory
            start_y, start_x = self.get_random_trajectory(self.n_frames_total, motion_type=self.motion_type)
            ind = random.randint(0, self.mnist.shape[0] - 1)
            digit_image = self.mnist[ind]
            for i in range(self.n_frames_total):
                top    = start_y[i]
                left   = start_x[i]
                bottom = top + self.digit_size_
                right  = left + self.digit_size_
                # Draw digit
                data[i, top:bottom, left:right] = np.maximum(data[i, top:bottom, left:right], digit_image)

        data = data[..., np.newaxis]
        return data

    def __getitem__(self, idx):
        length = self.n_frames_input + self.n_frames_output
        if self.is_train or self.num_objects[0] != 2:
            # Sample number of objects
            num_digits = self.num_objects
            # Generate data on the fly
            images = self.generate_moving_mnist(num_digits)
        else:
            images = self.dataset[:, idx, ...]

        if self.transform is not None:
            images = self.transform(images)
        input = images[:self.n_frames_input]
        if self.n_frames_output > 0:
            output = images[self.n_frames_input:length]
        else:
            output = []

        return input #, output

    def __len__(self):
        return len(self.split)