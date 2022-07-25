import h5py
import numpy as np
from torch.utils.data import Dataset
import os
from skimage import io

class TrainDataset(Dataset):
    def __init__(self, HR_dir, LR_dir):
        super(TrainDataset, self).__init__()
        self.HR_dir = HR_dir
        self.LR_dir = LR_dir

    def __getitem__(self, idx):
        all_images = os.listdir(self.HR_dir)
        HR_path, LR_path = os.path.join(self.HR_dir,all_images[idx]), os.path.join(self.LR_dir,all_images[idx])
        HR = io.imread(HR_path) / 255
        LR = io.imread(LR_path) / 255
        return LR, HR, imagename
        
    def __len__(self):
        all_images = os.listdir(self.HR_dir)
        return len(all_images)



