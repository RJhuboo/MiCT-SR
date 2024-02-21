import argparse
import os
import copy
import pickle
import torch
from torch import nn
from torch.nn import L1Loss, MSELoss
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from models import FSRCNN, BPNN
from datasets import TrainDataset
from utils import AverageMeter, calc_psnr

NB_DATA = 2800+4700

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--HR_dir', type=str,default = "./data/HR/Train_Label_trab_100")
    parser.add_argument('--LR_dir', type=str,default = "./data/LR/Train_trab")
    parser.add_argument('--mask_dir', type=str,default = "./data/HR/Train_trab_mask")
    parser.add_argument('--outputs-dir', type=str, default = "./FSRCNN_search")
    parser.add_argument('--checkpoint_bpnn', type= str, default = "./checkpoints_bpnn/BPNN_checkpoint_lrhr.pth")
    parser.add_argument('--alpha',type=float, default = 0.005)
    parser.add_argument('--Loss_bpnn', default = MSELoss)
    parser.add_argument('--weights-file', type=str)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=120)
    parser.add_argument('--num-workers', type=int, default=6)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--nof', type= int, default = 23)
    parser.add_argument('--n1', type=int,default = 158)
    parser.add_argument('--n2', type=int,default = 152)
    parser.add_argument('--n3', type=int,default = 83)
    parser.add_argument('--gpu_ids', type=list, default = [0,1,2])
    parser.add_argument('--NB_LABEL', type=int, default = 7)
    args = parser.parse_args()

    args.outputs_dir = os.path.join(args.outputs_dir, 'BPNN_alpha0_0001_TF_x{}'.format(args.scale))
    
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    
    # Prepare multigpus
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count())
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Load BPNN
    model_bpnn = BPNN(in_channel=1,features=args.nof, out_channels=args.NB_LABEL, n1= args.n1, n2=args.n2, n3=args.n3, k1=3,k2=3,k3=3).to(device)
    model_bpnn.load_state_dict(torch.load(os.path.join(args.checkpoint_bpnn)))
    if torch.cuda.device_count() > 1:
        model_bpnn = nn.DataParallel(model_bpnn)
    model_bpnn.to(device)
    for param in model_bpnn.parameters():
        param.requires_grad = False
    model_bpnn.eval()

                       
    score_bpnn, score_sr, score_psnr = [], [], []
    
    # Create FSRCNN model with manual seed 
    torch.manual_seed(args.seed)
    model = FSRCNN(scale_factor=args.scale)
    optimizer = optim.Adam([
                            {'params': model.first_part.parameters()},
                            {'params': model.mid_part.parameters()},
                            {'params': model.last_part.parameters(), 'lr': args.lr * 0.1}
                            ], lr=args.lr)
    if torch.cuda.device_count() >1:
        model = nn.DataParallel(model) 
    model.to(device)
    
    # Losses initialization
    criterion = nn.MSELoss()
    Lbpnn =  args.Loss_bpnn()
    
    # Dataset & Dataloader
    dataset = TrainDataset(args.HR_dir, args.LR_dir)
    train_dataloader = DataLoader(dataset=dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers)
                                  #pin_memory=True)  
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_loss = 10

    # Start epoch
    for epoch in range(args.num_epochs):
      
        model.train()
        epoch_losses = AverageMeter()
        bpnn_loss = AverageMeter()
        psnr_train = AverageMeter()
        with tqdm(total=(len(train_dataloader) - len(train_dataloader) % args.batch_size), ncols=80) as t:
            t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))
            
            # Start training
            for data in train_dataloader:
                
                # Forward
                inputs, labels, masks = data
                inputs = inputs.reshape(inputs.size(0),1,256,256)
                labels = labels.reshape(labels.size(0),1,512,512)
                masks = masks.reshape(masks.size(0),1,512,512)
                inputs, labels, masks = inputs.to(device), labels.to(device), masks.to(device)

                preds,preds_bin = model(inputs)
                preds = preds.clamp(0.0,1.0)

                # Initialization for label binarization
                labels_bin = labels.clone().detach()
                masks_bin = masks.clone().detach()
                masks_bin = F.interpolate(masks_bin, size=64)

                # Binarized the label image
                gaussian_blur = transforms.GaussianBlur((3,3),3)
                labels_bin = gaussian_blur(labels_bin)
                labels_bin = labels_bin.cpu().numpy()
                labels_bin = labels_bin>0.225 # Empirical Value, must be tuned or Labels_bin must be provided differently
                labels_bin = labels_bin.astype("float32")
                labels_bin = torch.from_numpy(labels_bin).to(device)

                # Compute Morphometric parameters
                P_SR = model_bpnn(masks_bin,preds_bin)
                P_HR = model_bpnn(masks_bin,labels_bin)
                BVTV_SR = bvtv_loss(preds_bin,masks)
                BVTV_HR = bvtv_loss(labels,masks)
                P_SR = torch.cat((P_SR,BVTV_SR),dim=1)
                P_HR = torch.cat((P_HR,BVTV_HR),dim=1)
                
                # Compute the loss
                L_SR = criterion(preds, labels)
                L_BPNN = Lbpnn(P_SR,P_HR)
                loss = L_SR + (args.alpha * L_BPNN)

                # Backward
                optimizer.zero_grad()
                loss.mean().backward()
                optimizer.step()

                # Performance
                epoch_losses.update(loss.item(), len(inputs))
                bpnn_loss.update(L_BPNN.item())
                with torch.no_grad():
                    psnr_train.update(calc_psnr(labels.cpu(),preds.clamp(0.0,1.0).cpu(),masks.cpu(),device="cpu").item())

                # Display the performance
                t.set_postfix(loss='{:.9f}'.format(epoch_losses.avg),LossSR='{:.9f}'.format(L_SR.item()),bpnn='{:.3f}'.format(bpnn_loss.avg),psnr='{:.1f}'.format(psnr_train.avg),ssim='{:.1f}'.format(ssim_train.avg),alpha='{:.8f}'.format(args.alpha[trial]))
                t.update(len(inputs))
                
  
        
        # Display result Losses
        print("##### Train #####")
        print("BPNN loss: {:.6f}".format(bpnn_loss.avg))
        print("train loss : {:.6f}".format(epoch_losses.avg))
        
        # Save model
        if torch.cuda.device_count() > 1:
            torch.save(model.module.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))
        else:
            torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))

    print('best epoch: {}, loss: {:.6f}'.format(best_epoch, best_loss))

