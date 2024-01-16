import numpy as np
from torch.utils.data import Dataset
import os
from skimage import io, transform
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
from PIL import Image

class TrainDataset(Dataset):
    def __init__(self, HR_dir, LR_dir, mask_dir,transform=None):
        super(TrainDataset, self).__init__()
        self.HR_dir = HR_dir
        self.LR_dir = LR_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __getitem__(self, idx):
        all_images = os.listdir(self.HR_dir)
        HR_path, LR_path = os.path.join(self.HR_dir,all_images[idx]), os.path.join(self.LR_dir,all_images[idx])
        mask_path = os.path.join(self.mask_dir, all_images[idx].replace(".png",".bmp"))
        HR = Image.open(HR_path)
        LR = Image.open(LR_path)
        LR = LR.resize((128,128))
        LR = np.array(LR) /255
        HR = np.array(HR) /255
        #HR = io.imread(HR_path) / 255
        #LR = io.imread(LR_path) / 255
        
        mask = (np.array(Image.open(mask_path))>0)*1.
        #mask = transform.rescale(mask, 1/8, anti_aliasing=False) / 255
        HR = HR.astype('float32')
        LR = LR.astype('float32')
        mask = mask.astype('float32')
        p = random.random()
        rot = random.randint(-45,45)
        transform_list = []
        HR,LR,mask=TF.to_pil_image(HR),TF.to_pil_image(LR),TF.to_pil_image(mask)
        HR,LR,mask=TF.rotate(HR,rot),TF.rotate(LR,rot),TF.rotate(mask,rot)
        if p<0.3:
            HR,LR,mask=TF.vflip(HR),TF.vflip(LR),TF.vflip(mask)
        p = random.random()
        if p<0.3:
            HR,LR,mask=TF.hflip(HR),TF.hflip(LR),TF.hflip(mask)
        p = random.random()
        if p>0.2:
            HR,LR,mask=TF.affine(HR,angle=0,translate=(0.1,0.1),shear=0,scale=1),TF.affine(LR,angle=0,translate=(0.1,0.1),shear=0,scale=1),TF.affine(mask,angle=0,translate=(0.1,0.1),shear=0,scale=1)
        HR,LR,mask=TF.to_tensor(HR),TF.to_tensor(LR),TF.to_tensor(mask)
        #if self.transform:
        #    HR = self.transform(HR)
        #    LR = self.transform(LR)
        #    mask = self.transform(mask)
        #print(np.unique(LR))
        #print(np.shape(LR))
        #print(LR,HR)
        return LR, HR, mask, all_images[idx]
        
    def __len__(self):
        all_images = os.listdir(self.HR_dir)
        return len(all_images)
    
class TestDataset(Dataset):
    def __init__(self, HR_dir, LR_dir, mask_dir=None):
        super(TestDataset, self).__init__()
        self.HR_dir = HR_dir
        self.LR_dir = LR_dir
        self.mask_dir = mask_dir
    def __getitem__(self, idx):
        all_images = os.listdir(self.HR_dir)
        HR_path, LR_path = os.path.join(self.HR_dir,all_images[idx]), os.path.join(self.LR_dir,all_images[idx])
        mask_path = os.path.join(self.mask_dir, all_images[idx].replace(".png",".bmp"))
        HR = Image.open(HR_path)
        LR = Image.open(LR_path)
        LR = LR.resize((128,128))
        LR = np.array(LR)/255
        HR = np.array(HR)/255
        mask = np.array(Image.open(mask_path))
        #HR = io.imread(HR_path) / 255
        #LR = io.imread(LR_path) / 255
        #mask = io.imread(mask_path) / 255
        HR = HR.astype('float32')
        LR = LR.astype('float32')
        mask = mask.astype('float32')
        return LR, HR, mask, all_images[idx]
    def __len__(self):
        all_images = os.listdir(self.HR_dir)
        return len(all_images)

