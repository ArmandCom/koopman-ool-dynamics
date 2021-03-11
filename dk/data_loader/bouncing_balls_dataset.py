from glob import glob
import os
import numpy as np
import random
import torch.utils.data as data
import json
import cv2

def make_dataset(root, is_train):
    if is_train:
        folder = 'balls_n4_t60_ex50000_m'
    else:
        folder = 'balls_n4_t60_ex2000_m'
    dataset = np.load(os.path.join(root, folder, 'dataset_info.npy'))
    return dataset

class BouncingBallsDataset(data.Dataset):
    '''
    Bouncing balls dataset.
    '''
    def __init__(self, root, train, n_frames_input, n_frames_output, num_objects, image_size,
                 train_split = 0.8, transform=None, return_positions=False):
        # (self, root, train, n_frames_input, n_frames_output, num_objects,
        #              train_split=0.8, transform=None):
        super(BouncingBallsDataset, self).__init__()
        self.n_frames = n_frames_input + n_frames_output
        self.dataset = make_dataset(root, train)
        self.size = image_size
        self.scale = [self.size[0] / 800, self.size[1] / 800]
        self.radius = int(60 * max(self.scale)) # Note: previously 60 *

        sam_rate = 2
        self.sam_rate = int(sam_rate)
        self.root = root
        self.is_train = train
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.transform = transform
        self.return_positions = return_positions

        self.colors = [(10, 255, 0),
                       (255, 10, 0),
                       (0, 0, 255),
                       (255, 50, 255),
                       (255, 255, 50),
                       (50, 255, 255),
                       (153, 255, 153)]
        assert len(self.colors) >= num_objects
        assert num_objects == self.dataset.shape[-2]

        #
        idx_full = np.arange(len(self.dataset))
        np.random.seed(0)
        np.random.shuffle(idx_full)
        self.split = idx_full

        # self.is_train = train
        # self.num_objects = num_objects
        # self.n_frames_input = n_frames_input
        # self.n_frames_output = n_frames_output
        # self.n_frames_total = self.n_frames_input + self.n_frames_output
        # self.transform = transform
        # # For generating data
        # self.image_size_ = image_size
        # self.step_length_ = 0.25

    def __getitem__(self, idx):
        # traj sizeL (n_frames, n_balls, 4)
        traj = self.dataset[idx]
        vid_len, n_balls = traj.shape[:2]
        if self.is_train:
            start = random.randint(0, vid_len - self.sam_rate*self.n_frames)
        else:
            start = 0

        n_channels = 3
        images = np.zeros([self.n_frames, self.size[0], self.size[1], n_channels], np.uint8)
        positions = []
        for fid in range(self.n_frames):
            xy = []
            for color, bid in zip(self.colors[:n_balls], range(n_balls)):
                # each ball:
                ball = traj[start + fid*self.sam_rate, bid]
                x, y = int(round(self.scale[0] * ball[0])), int(round(self.scale[1] * ball[1]))
                images[fid] = cv2.circle(images[fid], (x, y), int(self.radius * ball[3]),
                               color, -1)

                xy.append([x / self.size[0], y / self.size[1]])
            positions.append(xy)

        if self.transform is not None:
            images = self.transform(images)

        input = images[:self.n_frames_input]
        if self.n_frames_output > 0:
            output = images[self.n_frames_input:]
        else:
            output = []

        if not self.return_positions:
            return input#, output
        else:
            positions = np.array(positions)
            return input, positions #, output,

    def __len__(self):
        return len(self.dataset)