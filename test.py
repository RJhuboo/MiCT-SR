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
    parser.add_argument('--label-dir', type=str, required=True)
    parser.add_argument('--mask-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str,required=True)
    parser.add_argument('--checkpoint_bpnn', type= str, default = "./checkpoints_bpnn/BPNN_checkpoint_12p.pth")
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
    
    model_bpnn = BPNN(args.nof, args.NB_LABEL, n1= args.n1, n2=args.n2, n3=args.n3, k1=3,k2=3,k3=3).to(device)
    model_bpnn.load_state_dict(torch.load(os.path.join(args.checkpoint_bpnn)))
    model_bpnn.to(device)
    
    for param in model_bpnn.parameters():
        param.requires_grad = False
    model_bpnn.eval()
    
    model = FSRCNN(scale_factor=args.scale).to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()

    psnr = []
    ssim_list = []
    mse = []
## --- TESTING LOOP --- ##

    for image_file in os.listdir(args.image_dir):
        image = pil_image.open(os.path.join(args.image_dir,image_file))
        image2 = pil_image.open(os.path.join(args.label_dir,image_file))
        image_width = (image.width // args.scale) * args.scale
        image_height = (image.height // args.scale) * args.scale

        #hr = image2.resize((image_width, image_height), resample=pil_image.BICUBIC)
        hr = image2
        lr = image 
        #lr = image.resize((hr.width // args.scale, hr.height // args.scale),resample=pil_image.BICUBIC)
       
        lr = preprocess(lr, device)
        hr= preprocess(hr, device)
        
        #savelr = preprocess(image,device)

        with torch.no_grad():
            preds = model(lr).clamp(0.0, 1.0)
        
            P_SR = model_bpnn(preds)
            P_HR = model_bpnn(hr)
            Loss = MSELoss()
            mse.append(Loss(P_SR,P_HR).item())
        
        print(hr.size())
        print(preds.size())
        psnr.append(calc_psnr(hr, preds, args.mask_dir, image_file,device))
        ssim_list.append(ssim(x=hr, y=preds, data_range=1., downsample = False, directory=args.mask_dir, maskname=image_file,device=device))
        #psnr2.append(calc_psnr(hr,savelr))

        preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

        output = np.array(preds)
        output = np.clip(output, 0.0, 255.0).astype(np.uint8)
        output = pil_image.fromarray(output)
        directory = args.output_dir
        name_save = image_file  
        output.save(os.path.join(directory, name_save))

    print('PSNR COMPUTATION ...')
    print('Average PSNR SR :', sum(psnr)/len(psnr))
    print('Average MSE :', sum(mse)/len(mse))
    print('Average SSIM SR :', sum(ssim_list)/len(ssim_list))
    #print('Average PSNR LR : ', sum(psnr2)/len(psnr2))
