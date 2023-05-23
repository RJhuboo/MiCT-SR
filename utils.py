import torch
from torch import nn
import numpy as np
from skimage import io
import os
import pytorch_ssim
import torch.autograd as autograd
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

def local_thickness(img,mask=None,voxel_size = 10.5,sep=True):
    # Convert input tensors to NumPy arrays
    #img_np = img.cpu().numpy()
    #mask_np = mask.cpu().numpy() if mask is not None else None
    if sep:
        img = np.logical_not(img)
    
    # Compute the skeleton and distance transform of the binary image using medial axis
    skel, dist = morphology.medial_axis(img, mask=mask, return_distance=True)

    thickness_skeleton = np.ma.masked_array(dist, np.logical_not(skel).astype(bool))
    mean_thickness = (
        (np.sum(thickness_skeleton.compressed()) * 2) /
        np.sum(np.ma.masked_array(skel, np.logical_not(mask).astype(bool)) * 1)
    ) * voxel_size
    
    # Convert the mean_thickness to a PyTorch tensor and return
    #mean_thickness = torch.from_numpy(np.array(mean_thickness)).float()
    return mean_thickness
#def local_thickness(img, mask=None, voxel_size=10.5, sep=True):
    # Invert the image if sep=True
#    if sep:
#        img = torch.logical_not(img)

    # Compute the skeleton and distance transform of the binary image using medial axis
#    skel, dist = skel, dist = morphology.medial_axis(img, mask=mask, return_distance=True)
#    thickness_skeleton = torch.where(skel, dist, torch.tensor(float('inf'), device=img.device))
#    mean_thickness = (
#        (torch.sum(thickness_skeleton) * 2) /
#        torch.sum(torch.logical_and(skel, mask)) * 1
#    ) * voxel_size
    
#### Trabecular pattern factor function  ####

def Trabecular_pattern(img,pixel_size = 10.5):

    #img_np = img.cpu().numpy()
    # Smooth the trabecular bone surfaces
    smoothed = morphology.binary_closing(img, morphology.disk(2))
    smoothed = torch.as_tensor(smoothed, dtype=torch.uint8) #util.img_as_ubyte(smoothed)

    # Dilate the trabeculae by a rank order operator (median filter)
    dilated = filters.rank.median(smoothed*255, morphology.disk(1))


    # Measure bone area (Al) and bone perimeter (Pl) of the binary image
    props1 = measure.regionprops(smoothed,spacing=(pixel_size,pixel_size))
    Al = props1[0].area
    Pl = props1[0].perimeter 


    # Measure bone area (A2) and bone perimeter (P2) of the dilated image
    props2 = measure.regionprops(dilated,spacing=(pixel_size,pixel_size))
    A2 = props2[0].area
    P2 = props2[0].perimeter  

    # Calculate TBPf as the quotient of the differences of the first and the second measurement
    TBPf = (A2 - Al) / (P2 - Pl)


    return TBPf

#### Equivalent circle diameter function ####

#def Equivalent_circle_diameter(img,pixel_size=10.5):

    # Label the objects of the image 
#    labeled = morphology.label(img)

    # Diameter initilization
#    DEA = 0
#    count = 0

    # Loop into all the objects of the image 
#    for region in measure.regionprops(labeled,spacing=pixel_size):

        # Measure bone circle diameter equivalent area 
 #       DEA += region.equivalent_diameter_area
 #       count += 1
 #   return DEA/count
def Equivalent_circle_diameter(img, pixel_size=10.5):
    # Label the objects of the image
    labeled = morphology.label(img)

    # Diameter initialization
    DEA = torch.tensor(0, dtype=torch.float32)
    count = torch.tensor(0, dtype=torch.float32)

    # Loop into all the objects of the image
    for region in measure.regionprops(labeled):
        # Measure bone circle diameter equivalent area
        DEA += torch.tensor(region.equivalent_diameter_area * 10.5, dtype=torch.float32)
        count += 1

    return DEA / count
####  mixed function of Bone perimeter/area ratio, area, number of objects ####

#def perimeter_area(img,pixel_size=10.5):

    # Label the objects of the image 
#    labeled = morphology.label(img)

#    # Diameter initilization
#    area = 0
#    perimeter_area_ratio = 0
#    count = 0

#    # Loop into all the objects of the image 
#    for region in measure.regionprops(labeled):#spacing=pixel_size):
#
#        # Measure bone circle diameter equivalent area 
#        area += region.area
#        perimeter_area_ratio += region.perimeter / (region.area*pixel_size)
#        count += 1
#    return (area*(pixel_size**2))/count, perimeter_area_ratio/count, count
def perimeter_area(img, pixel_size=10.5):
    # Label the objects of the image
    labeled = morphology.label(img)

    # Initialization
    area = torch.tensor(0, dtype=torch.float32)
    perimeter_area_ratio = torch.tensor(0, dtype=torch.float32)
    count = torch.tensor(0, dtype=torch.float32)

    # Loop into all the objects of the image
    for region in measure.regionprops(labeled):
        # Measure area and perimeter-area ratio
        area += torch.tensor(region.area, dtype=torch.float32)
        perimeter_area_ratio += torch.tensor(region.perimeter) / (torch.tensor(region.area) * pixel_size)
        count += 1

    return (area * (pixel_size ** 2)) / count, perimeter_area_ratio / count, count
####  Bone to total area ratio ####

#def BVTV(img,mask):
#    BV=np.count_nonzero(img)
#    TV=np.count_nonzero(mask)
#    return (BV/TV) * 100
def BVTV(img, mask):
    BV = torch.count_nonzero(img)
    TV = torch.count_nonzero(mask)
    return (BV / TV) * 100

class MorphLoss(autograd.Function):
    @staticmethod
    def forward(ctx,img,mask,target,voxel_size):
        ctx.save_for_backward(img, target)
        ctx.mask = mask
        ctx.voxel_size = voxel_size
        img = img.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy()
        batch_size = img.shape[0]
        target = target.detach().cpu().numpy()
        img = (img[0,0,:,:]>0.22)*1
        mask = mask[0,0,:,:]>0.5
        target = (target[0,0,:,:]>0.22)*1
        thck_h=np.zeros((batch_size))
        thck_l=np.zeros((batch_size))
        for batch in batch_size:
            thicknessl = local_thickness(img[batch_size,:,:,:],mask[batch_size,:,:,:],voxel_size,sep=False)
            thicknessh = local_thickness(target[batch_size,:,:,:],mask[batch_size,:,:,:],voxel_size,sep=False)
            #bvtvl = BVTV(img,mask)
            #bvtvh = BVTV(target,mask)
            #areal, perimeterl, nbobjl = perimeter_area(img,self.voxel_size)
            #areah, perimeterh, nbobjh = perimeter_area(target,self.voxel_size)
            #diameterl = Equivalent_circle_diameter(img,self.voxel_size)
            #diameterh = Equivalent_circle_diameter(target,self.voxel_size)
            #separationl = local_thickness(img,mask,self.voxel_size,sep=True)
            #separationh = local_thickness(target,mask,self.voxel_size,sep=True)
            thck_h[batch]=(thicknessl-48.7578)/5.2874
            thck_l[batch]=(thicknessh-48.7578)/5.2874
        
        #loss2 = self.loss((bvtvl-16.1473)/7.64596,(bvtvh-16.1473)/7.64596)
        #loss3 = self.loss((areal-11166.6)/6145.2,(areah-11166.6)/6145.2)
        #loss4 = self.loss((perimeterl-0.0583277)/0.00581525,(perimeterh-0.0583277)/0.00581525)
        #loss5 = self.loss((nbobjl-25.8531)/12.2747,(nbobjh-25.8531)/12.2747)
        #loss6 = self.loss((diameterl-30.2512)/6.56558,(diameterh-30.2512)/6.56558)
        #loss7 = self.loss((separationl-156.729)/46.1809,(separationh-156.729)/46.1809)
        # MSE 
        loss1 = torch.square(torch.sum(torch.tensor(thck_h) - torch.tensor(thck_l)))/(batch_size * 2)
        return loss1
    
    @staticmethod
    def backward(ctx, grad_output):
        input_img, label_img = ctx.saved_tensors
        mask = ctx.mask
        voxel_size = ctx.voxel_size

        # Convert torch tensors to NumPy arrays
        input_img_np = input_img.detach().cpu().numpy()
        label_img_np = label_img.detach().cpu().numpy()
        
        batch_size = input_img_np.shape[0]
        thck_h=np.zeros((batch_size))
        thck_l=np.zeros((batch_size))
        
        for batch in batch_size:
            thicknessl = local_thickness(input_img_np[batch,:,:,:],mask[batch,:,:,:],voxel_size,sep=False)
            thicknessh = local_thickness(label_img_np[batch,:,:,:],mask[batch_size,:,:,:],voxel_size,sep=False)
            thck_h[batch]=(thicknessl-48.7578)/5.2874
            thck_l[batch]=(thicknessh-48.7578)/5.2874

        # Compute the gradients of the input image using chain rule
        grad_input = (1/batch_size)*torch.sum(torch.tensor(thck_h) - torch.tensor(thck_l))*grad_output
        return grad_input

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
