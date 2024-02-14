import argparse

import torch
import os
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
from torch.nn import L1Loss, MSELoss

from models import FSRCNN, BPNN
from utils import convert_ycbcr_to_rgb, preprocess, calc_psnr
from ssim import ssim

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--image-dir', type=str, required=True)
    parser.add_argument('--nof', type= int, default = 23)
    parser.add_argument('--n1', type=int,default = 169)
    parser.add_argument('--n2', type=int,default = 155)
    parser.add_argument('--n3', type=int,default = 154)
    parser.add_argument('--NB_LABEL', type=int,default=12)
    parser.add_argument('--scale', type=int, default=2)
    args = parser.parse_args()

## ---  INITIALISATION STEP --- ##

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = FSRCNN(scale_factor=args.scale).to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()
    
    ## --- TESTING LOOP --- ##

    for image_file in os.listdir(args.image_dir):
        image = pil_image.open(os.path.join(args.image_dir,image_file))
        image_width = (image.width // args.scale) * args.scale
        image_height = (image.height // args.scale) * args.scale

        lr = image 
       
        lr = preprocess(lr, device)
        with torch.no_grad():
            preds = model(lr).clamp(0.0, 1.0)
        
        preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

        output = np.array(preds)
        output = np.clip(output, 0.0, 255.0).astype(np.uint8)
        output = pil_image.fromarray(output)
        directory = args.output_dir
        name_save = image_file  
        output.save(os.path.join(directory, name_save))
