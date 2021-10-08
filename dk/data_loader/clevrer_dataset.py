from glob import glob
import torchvision
import os
import numpy as np
import random
import torch.utils.data as data
import json
import cv2
from tqdm import tqdm

def generate_dataset(root, is_train, batch_size=1, start_idx=0):
    if is_train:
        folder = 'video_train'
    else:
        folder = 'video_validation'

    video_dirs = []
    for subdir, dirs, files in os.walk(os.path.join(root, folder)):
        for subdir, dirs, files in os.walk(os.path.join(root, folder, subdir)):
            for file in files:
                video_dirs.append(os.path.join(root, folder, subdir, file))

    frames = []
    for i, video_dir in enumerate(tqdm(video_dirs)):
        if i < start_idx:
            continue
        cap = cv2.VideoCapture(video_dir)
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                frames.append(frame)
            else:
                np.savez(os.path.join(root, folder + '_npz/', str(i).zfill(5)),
                         video=np.stack(frames))
                frames = []
                break
    return video_dirs

def generate_dataset_list(root, is_train):
    if is_train:
        folder = 'video_train'
    else:
        folder = 'video_validation'

    video_dirs = []
    for subdir, dirs, files in os.walk(os.path.join(root, folder)):
        for subdir, dirs, files in os.walk(os.path.join(root, folder, subdir)):
            for file in files:
                video_dirs.append(os.path.join(root, folder, subdir, file))

    # for i, video_dir in enumerate(tqdm(video_dirs)):
    #     video = torchvision.io.read_video(video_dir)[0]

    return video_dirs

    # Option 2
    # frames = []
    # frames_batched = []
    # count = 0
    # for i, video_dir in enumerate(tqdm(video_dirs)):
    #     cap = cv2.VideoCapture(video_dir)
    #     while(cap.isOpened()):
    #         ret, frame = cap.read()
    #         if ret == True:
    #             frames.append(frame)
    #         else:
    #             frames_batched.append(np.stack(frames))
    #             frames = []
    #             break
    #     if (i+1) % batch_size == 0:
    #         np.savez(os.path.join(root, folder + '_npz/', str(count).zfill(5)),
    #                  video=np.stack(frames_batched))
    #         # frame_array.append(np.array(frames))
    #         frames_batched = []
    #         count += 1

class ClevrerDataset(data.Dataset):
    '''
    Bouncing balls dataset.
    '''
    def __init__(self, root, train, n_frames_input, n_frames_output, num_objects, image_size,
                 train_split = 0.8, transform=None, return_positions=False):
        super(ClevrerDataset, self).__init__()
        self.root = root
        self.is_train = train
        self.n_frames = n_frames_input + n_frames_output
        self.len_video = 128
        # generate_dataset(root, train, start_idx = 5728)
        self.dataset_list = generate_dataset_list(root, train)
        self.num_videos = self.get_length()
        if self.num_videos == 0:
            self.dataset_list = generate_dataset_list(root, train)
            self.num_videos = self.get_length()

        self.clips_x_video = self.len_video // self.n_frames
        self.length = self.num_videos

        sam_rate = 4
        self.sam_rate = int(sam_rate)

        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.transform = transform

        # assert len(self.colors) >= num_objects
        # assert num_objects == self.dataset.shape[-2]

        idx_full = np.arange(self.num_videos)
        np.random.seed(0)
        np.random.shuffle(idx_full)
        self.split = idx_full

    def get_video(self, idx):
        if self.is_train:
            folder = 'video_train'
        else:
            folder = 'video_validation'

        # Option 1: load from npz (not all of them are saved)
        # video = np.load(os.path.join(self.root, folder + '_npz', str(idx).zfill(5) + '.npz'))['video']

        # Option 2: load from mp4 directly
        video = torchvision.io.read_video(self.dataset_list[idx])[0]

        return video

    def get_length(self):
        if self.is_train:
            folder = 'video_train'
        else:
            folder = 'video_validation'
        length = 0
        # Option 1: Load from npz
        # for subdir, dirs, files in os.walk(os.path.join(self.root, folder + '_npz')):
        #     length += len(files)

        # Option 2: Load from mp4
        length = len(self.dataset_list)
        return length

    def __getitem__(self, idx):
        # traj sizeL (n_frames, n_balls, 4)
        video = self.get_video(idx)
        T, H, W, C = video.shape
        if self.is_train:
            start = random.randint(0, T - self.sam_rate*self.n_frames)
        else:
            start = 0
            # TODO: we are throwing T - self.sam_rate*self.n_frames frames
            #  for validation and test. So this is provisional

        images = video[start:start+self.n_frames*self.sam_rate:self.sam_rate]
        if self.transform is not None:
            images = self.transform(images)

        input = images[:self.n_frames_input]
        if self.n_frames_output > 0:
            output = images[self.n_frames_input:]
        else:
            output = []
        return input

    def __len__(self):
        return self.num_videos