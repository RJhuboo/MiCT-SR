import numpy as np
from torch.utils.data import Dataset
import os
from skimage import io, transform
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random

class TrainDataset(Dataset):
    def __init__(self, HR_dir,HR_bin_dir, LR_dir, mask_dir,transform=None):
        super(TrainDataset, self).__init__()
        self.HR_dir = HR_dir
        self.LR_dir = LR_dir
        self.mask_dir = mask_dir
        self.HR_bin_dir = HR_bin_dir
        self.transform = transform

    def __getitem__(self, idx):
        all_images = os.listdir(self.HR_dir)
        HR_path, LR_path, HR_bin_path = os.path.join(self.HR_dir,all_images[idx]), os.path.join(self.LR_dir,all_images[idx]),os.path.join(self.HR_bin_dir,all_images[idx])
        mask_path = os.path.join(self.mask_dir, all_images[idx].replace(".png",".bmp"))
        HR = io.imread(HR_path) / 255
        HR_bin = io.imread(HR_bin_path) / 255
        LR = io.imread(LR_path) / 255
        mask = io.imread(mask_path) /255
        #mask = transform.rescale(mask, 1/8, anti_aliasing=False) / 255
        HR = HR.astype('float32')
        HR_bin = HR_bin.astype('float32')
        LR = LR.astype('float32')
        mask = mask.astype('float32')
        
        p = random.random()
        rot = random.randint(-45,45)
        transform_list = []
        HR,HR_bin,LR,mask=TF.to_pil_image(HR),TF.to_pil_image(HR_bin),TF.to_pil_image(LR),TF.to_pil_image(mask)
        HR,HR_bin,LR,mask=TF.rotate(HR,rot),TF.rotate(HR_bin,rot),TF.rotate(LR,rot),TF.rotate(mask,rot)
        if p<0.3:
            HR,HR_bin,LR,mask=TF.vflip(HR),TF.vflip(HR_bin),TF.vflip(LR),TF.vflip(mask)
        p = random.random()
        if p<0.3:
            HR,HR_bin,LR,mask=TF.hflip(HR),TF.hflip(HR_bin),TF.hflip(LR),TF.hflip(mask)
        p = random.random()
        if p>0.2:
            HR,HR_bin,LR,mask=TF.affine(HR,angle=0,translate=(0.1,0.1),shear=0,scale=1),TF.affine(HR_bin,angle=0,translate=(0.1,0.1),shear=0,scale=1),TF.affine(LR,angle=0,translate=(0.1,0.1),shear=0,scale=1),TF.affine(mask,angle=0,translate=(0.1,0.1),shear=0,scale=1)
        HR,HR_bin,LR,mask=TF.to_tensor(HR),TF.to_tensor(HR_bin),TF.to_tensor(LR),TF.to_tensor(mask)
        #if self.transform:
        #    HR = self.transform(HR)
        #    LR = self.transform(LR)
        #    mask = self.transform(mask)
        return LR, HR,HR_bin, mask, all_images[idx]
        
    def __len__(self):
        all_images = os.listdir(self.HR_dir)
        return len(all_images)
    
class TestDataset(Dataset):
    def __init__(self, HR_dir,HR_bin_dir LR_dir, mask_dir=None):
        super(TestDataset, self).__init__()
        self.HR_dir = HR_dir
        self.HR_bin_dir = HR_bin_dir
        self.LR_dir = LR_dir
        self.mask_dir = mask_dir
    def __getitem__(self, idx):
        all_images = os.listdir(self.HR_dir)
        HR_path,HR_bin_path LR_path = os.path.join(self.HR_dir,all_images[idx]),os.path.join(self.HR_bin_dir,all_images[idx]), os.path.join(self.LR_dir,all_images[idx])
        mask_path = os.path.join(self.mask_dir, all_images[idx].replace(".png",".bmp"))
        HR = io.imread(HR_path) / 255
        HR_bin = io.imread(HR_bin_path) / 255
        LR = io.imread(LR_path) / 255
        mask = io.imread(mask_path) / 255
        HR = HR.astype('float32
        HR_bin = HR_bin.astype('float32')
        LR = LR.astype('float32')
        mask = mask.astype('float32')
        return LR, HR,HR_bin, mask, all_images[idx]
    def __len__(self):
        all_images = os.listdir(self.HR_dir)
        return len(all_images)

