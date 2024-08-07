import sys, os
import imageio
import numpy as np
import torch
import torchvision

from torch.utils.data.dataset import Dataset


class SAV(Dataset):
    def __init__(self, root, mode, transform=None):
        self.root = root
        self.mode = mode
        self.transform = transform
        self.videos = []
        self.load_videos()
        
    def load_videos(self):
        for video in os.listdir(self.root):
            self.videos.append(video)
        
    def __getitem__(self, index):
        video = self.videos[index]
        video_path = os.path.join(self.root, video)
        frames = []
        for frame in os.listdir(video_path):
            frame_path = os.path.join(video_path, frame)
            img = imageio.imread(frame_path)
            frames.append(img)
        frames = np.array(frames)
        if self.transform is not None:
            frames = self.transform(frames)
        return frames
    
    def __len__(self):
        return len(self.videos)