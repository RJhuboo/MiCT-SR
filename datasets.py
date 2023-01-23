import numpy as np
from torch.utils.data import Dataset
import os
from skimage import io
import torchvision.transforms as transforms

class TrainDataset(Dataset):
    def __init__(self, HR_dir, LR_dir, mask_dir=None,transform=None):
        super(TrainDataset, self).__init__()
        self.HR_dir = HR_dir
        self.LR_dir = LR_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __getitem__(self, idx):
        all_images = os.listdir(self.HR_dir)
        HR_path, LR_path = os.path.join(self.HR_dir,all_images[idx]), os.path.join(self.LR_dir,all_images[idx])
        mask_path = os.path.join(self.mask_dir, all_images[idx].replace(".png",".bmp"))
        HR = io.imread(HR_path) / 255
        LR = io.imread(LR_path) / 255
        mask = io.imread(mask_path) / 255
        HR = HR.astype('float32')
        LR = LR.astype('float32')
        mask = mask.astype('float32')
        if self.transform:
            HR = self.transform(HR)
            LR = self.transform(LR)
            mask = self.transform(mask)
        return LR, HR, mask, all_images[idx]
        
    def __len__(self):
        all_images = os.listdir(self.HR_dir)
        return len(all_images)
    
    



