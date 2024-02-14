import torch
from torch import nn
import numpy as np
from skimage import io
import os
import pytorch_ssim
import torch.nn.functional as F
from skimage import io, filters, morphology, measure, feature, util
from scipy.ndimage import distance_transform_edt

def calc_patch_size(func):
    def wrapper(args):
        if args.scale == 2:
            args.patch_size = 10
        elif args.scale == 3:
            args.patch_size = 7
        elif args.scale == 4:
            args.patch_size = 6
        else:
            raise Exception('Scale Error', args.scale)
        return func(args)
    return wrapper


def convert_rgb_to_y(img, dim_order='hwc'):
    if dim_order == 'hwc':
        return 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256.
    else:
        return 16. + (64.738 * img[0] + 129.057 * img[1] + 25.064 * img[2]) / 256.


def convert_rgb_to_ycbcr(img, dim_order='hwc'):
    if dim_order == 'hwc':
        y = 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256.
        cb = 128. + (-37.945 * img[..., 0] - 74.494 * img[..., 1] + 112.439 * img[..., 2]) / 256.
        cr = 128. + (112.439 * img[..., 0] - 94.154 * img[..., 1] - 18.285 * img[..., 2]) / 256.
    else:
        y = 16. + (64.738 * img[0] + 129.057 * img[1] + 25.064 * img[2]) / 256.
        cb = 128. + (-37.945 * img[0] - 74.494 * img[1] + 112.439 * img[2]) / 256.
        cr = 128. + (112.439 * img[0] - 94.154 * img[1] - 18.285 * img[2]) / 256.
    return np.array([y, cb, cr]).transpose([1, 2, 0])


def convert_ycbcr_to_rgb(img, dim_order='hwc'):
    if dim_order == 'hwc':
        r = 298.082 * img[..., 0] / 256. + 408.583 * img[..., 2] / 256. - 222.921
        g = 298.082 * img[..., 0] / 256. - 100.291 * img[..., 1] / 256. - 208.120 * img[..., 2] / 256. + 135.576
        b = 298.082 * img[..., 0] / 256. + 516.412 * img[..., 1] / 256. - 276.836
    else:
        r = 298.082 * img[0] / 256. + 408.583 * img[2] / 256. - 222.921
        g = 298.082 * img[0] / 256. - 100.291 * img[1] / 256. - 208.120 * img[2] / 256. + 135.576
        b = 298.082 * img[0] / 256. + 516.412 * img[1] / 256. - 276.836
    return np.array([r, g, b]).transpose([1, 2, 0])


def preprocess(img, device):
    img = np.array(img).astype(np.float32)
    x = img
    #ycbcr = convert_rgb_to_ycbcr(img)
    #x = ycbcr[..., 0]
    x /= 255.
    x = torch.from_numpy(x).to(device)
    x = x.unsqueeze(0).unsqueeze(0)
    return x

####  Bone to total area ratio ####
def BVTV(img, mask):
    BV = torch.count_nonzero(img)
    TV = torch.count_nonzero(mask)
    return (BV / TV) * 100

class MorphLoss(nn.Module):
    def __init__(self,voxel_size=10.5):
        super(MorphLoss, self).__init__()
        self.voxel_size = voxel_size
        self.loss = nn.MSELoss()
    
    def forward(self,img,mask,target):
        img = img.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        img = (img[0,0,:,:]>0.22)*1
        mask = mask[0,0,:,:]>0.5
        target = (target[0,0,:,:]>0.22)*1
        thicknessl = local_thickness(img,mask,self.voxel_size,sep=False)
        thicknessh = local_thickness(target,mask,self.voxel_size,sep=False)
        #bvtvl = BVTV(img,mask)
        #bvtvh = BVTV(target,mask)
        #areal, perimeterl, nbobjl = perimeter_area(img,self.voxel_size)
        #areah, perimeterh, nbobjh = perimeter_area(target,self.voxel_size)
        #diameterl = Equivalent_circle_diameter(img,self.voxel_size)
        #diameterh = Equivalent_circle_diameter(target,self.voxel_size)
        separationl = local_thickness(img,mask,self.voxel_size,sep=True)
        separationh = local_thickness(target,mask,self.voxel_size,sep=True)
        loss1 = self.loss((thicknessl-48.7578)/5.2874,(thicknessh-48.7578)/5.2874)
        #loss2 = self.loss((bvtvl-16.1473)/7.64596,(bvtvh-16.1473)/7.64596)
        #loss3 = self.loss((areal-11166.6)/6145.2,(areah-11166.6)/6145.2)
        #loss4 = self.loss((perimeterl-0.0583277)/0.00581525,(perimeterh-0.0583277)/0.00581525)
        #loss5 = self.loss((nbobjl-25.8531)/12.2747,(nbobjh-25.8531)/12.2747)
        #loss6 = self.loss((diameterl-30.2512)/6.56558,(diameterh-30.2512)/6.56558)
        loss7 = self.loss((separationl-156.729)/46.1809,(separationh-156.729)/46.1809)
        print(loss1,loss7)
        total_loss = (1/2)*(loss1+loss7)
        return total_loss

#### PSNR Computation #### 
        
def calc_psnr(img1,img2,mask,device):
    MSE = torch.sum((torch.mul((img1 - img2),mask))**2) / torch.sum(mask)
    return 10. * torch.log10(1. / MSE)


#def calc_ssim(img1,img2, directory, name):
#    name = name[0].replace("png","bmp")
#    mask = io.imread(os.path.join(directory,name))
#    mask = mask / mask.max()
#    img1 = img1.cpu() * mask
#    img2 = img2.cpu() * mask
#    return pytorch_ssim.ssim(img1,img2).cpu().detach().numpy()


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
