from torch.autograd import Variable
import argparse
import os
import copy
import pickle
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import L1Loss, MSELoss
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
import torchvision
import torchvision.transforms as transforms
from skimage.filters import threshold_otsu 
import pytorch_ssim
from tqdm import tqdm
import optuna
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from models import FSRCNN, BPNN
from datasets import TrainDataset, TestDataset
from utils import AverageMeter, calc_psnr
from ssim import ssim
import pandas as pd
import time
import sys
import psutil
from PIL import Image
import math

NB_DATA = 7100
def bvtv_loss(tensor_to_count,tensor_mask):
    tensor_to_count = (tensor_to_count>0.2).type(torch.float32)*1.
    ones = (tensor_to_count == 1).sum(dim=2)
    BV = ones.sum(dim=2)
    ones = (tensor_mask == 1).sum(dim=2)
    TV = ones.sum(dim=2)
    BVTV = BV/TV
    return BVTV

def objective(trial):
    parser = argparse.ArgumentParser()
    parser.add_argument('--HR_dir', type=str,default = "./data/HR/Train_Label_trab_100")
    parser.add_argument('--LR_dir', type=str,default = "./data/LR/Train_trab")
    parser.add_argument('--mask_dir',type=str,default = "./data/HR/Train_trab_mask")
    parser.add_argument('--outputs-dir', type=str, default = "./FSRCNN_search")
    parser.add_argument('--checkpoint_bpnn', type= str, default = "./checkpoints_bpnn/BPNN_checkpoint_lrhr.pth")
    parser.add_argument('--alpha', type = list, default = [10,20,30,40,50,60])
    parser.add_argument('--Loss_bpnn', default = MSELoss)
    parser.add_argument('--weights-file', type=str)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)#-2
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--num-workers', type=int, default=24)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--nof', type= int, default = 64)
    parser.add_argument('--n1', type=int,default = 158)
    parser.add_argument('--n2', type=int,default = 152)
    parser.add_argument('--n3', type=int,default = 83)
    parser.add_argument('--gpu_ids', type=list, default = [0, 1, 2])
    parser.add_argument('--NB_LABEL', type=int, default = 7)
    parser.add_argument('--k_fold', type=int, default = 1)
    parser.add_argument('--name', type=str, default = "FSRCNN_Kbar_limit")
    args = parser.parse_args()

    args.outputs_dir = os.path.join(args.outputs_dir, args.name)    
    if os.path.exists(args.outputs_dir) == False:
        os.makedirs(args.outputs_dir)
    if os.path.exists("save_image/alpha_"+str(args.alpha[trial])) == False:
        os.makedirs("save_image/alpha_"+str(args.alpha[trial]))
    cudnn.benchmark = True
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count())
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model_bpnn = BPNN(in_channel=1,features=args.nof, out_channels=args.NB_LABEL, n1= args.n1, n2=args.n2, n3=args.n3, k1=3,k2=3,k3=3).to(device)
    model_bpnn.load_state_dict(torch.load(os.path.join(args.checkpoint_bpnn)))
    if torch.cuda.device_count() > 1:
        model_bpnn = nn.DataParallel(model_bpnn)
    model_bpnn.to(device)
    for param in model_bpnn.parameters():
        param.requires_grad = False
    model_bpnn.eval()
       
    #train_dataset = TrainDataset(args.HR_dir,args.LR_dir)
    index = range(NB_DATA)
    if args.k_fold >1:
        kf = KFold(n_splits = args.k_fold, shuffle=True)
    else:
        index = list(range(71))
        kf = train_test_split(index,train_size=args.alpha[trial],random_state=42)
        kf[0] = sorted(kf[0])
        new_kf = [list(range(kf[0][i]*100,(kf[0][i]+1)*100)) for i in range(args.alpha[trial])]
        new_kf=np.array(new_kf)
        kf[0] = np.vstack(new_kf).reshape((args.alpha[trial]*100,1)).flatten()
        new_kf = [list(range(kf[1][i]*100,(kf[1][i]+1)*100)) for i in range(71 - args.alpha[trial])]
        new_kf = np.array(new_kf)
        kf[1] = np.vstack(new_kf).reshape(((71-args.alpha[trial])*100,1)).flatten()
        #kf = train_test_split(index,train_size=6000,test_size=1100,shuffle=False)
    #cross_bpnn, cross_score, cross_psnr, cross_ssim = np.zeros(args.num_epochs),np.zeros(args.num_epochs),np.zeros(args.num_epochs),np.zeros(args.num_epochs)
    #cross_bpnn_train, cross_score_train, cross_psnr_train, cross_ssim_train = np.zeros(args.num_epochs),np.zeros(args.num_epochs),np.zeros(args.num_epochs),np.zeros(args.num_epochs)
    for k in range(1):
    # for train_index, test_index in kf.split(index):
        train_index = shuffle(kf[0])
        test_index = shuffle(kf[1])
        print("-------  Data separation -------")
        print("train size:",len(train_index))
        print("test size:",len(test_index))
        torch.manual_seed(args.seed)
        model = FSRCNN(scale_factor=args.scale,device=device)
        optimizer = optim.Adam([
                                {'params': model.first_part.parameters()},
                                {'params': model.mid_part.parameters()},
                                {'params': model.last_part.parameters(), 'lr': args.lr * 0.1}
                                ], lr=args.lr)
        #optimizer = optim.Adam(list(model_bpnn.parameters()) + list(model.parameters()))
        if torch.cuda.device_count() >1:
            model = nn.DataParallel(model) 
        model.to(device)
        criterion = nn.MSELoss()
        Lbpnn =  args.Loss_bpnn()
        
        my_transforms=None

        dataset = TrainDataset(args.HR_dir, args.LR_dir, args.mask_dir,transform = my_transforms)
        dataset_test = TestDataset("./data/HR/Test_trab","./data/LR/Test_trab","./data/HR/Test_mask")
        train_dataloader = DataLoader(dataset=dataset,
                                      batch_size=args.batch_size,
                                      sampler=train_index,
                                      num_workers=args.num_workers)
        print(len(train_dataloader))
        eval_dataloader = DataLoader(dataset=dataset, 
                                     sampler=test_index,
                                     batch_size=1,
                                     num_workers=args.num_workers)
        test_dataloader = DataLoader(dataset=dataset_test,batch_size=1,num_workers=args.num_workers)
        print(len(eval_dataloader))
        print(len(test_dataloader))
        best_weights = copy.deepcopy(model.state_dict())
        best_epoch = 0
        best_loss = 10
        tr_psnr = []
        tr_ssim = []
        tr_score,tr_bpnn = [], []
        t_score, t_bpnn,t_psnr,t_ssim = [],[],[],[]
        
        best_epoch_tracking = 1000
        data_param_SR = []
        data_param_HR = []
        data_param_SR_test = []
        data_param_HR_test = []
        names_index = []
        names_index_test = []
        for epoch in range(args.num_epochs):
            model.train()
            epoch_losses = AverageMeter()
            bpnn_loss = AverageMeter()
            psnr_train = AverageMeter()
            ssim_train = AverageMeter()
            with tqdm(total=(len(train_dataloader) - len(train_dataloader) % args.batch_size), ncols=80) as t:
                t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

                for data in train_dataloader:
                    inputs, labels, masks, imagename = data
                    inputs = inputs.reshape(inputs.size(0),1,256,256)
                    labels = labels.reshape(labels.size(0),1,512,512)
                    masks = masks.reshape(masks.size(0),1,512,512)
                    inputs, labels, masks = inputs.to(device), labels.to(device), masks.to(device)
                    
                    preds,preds_bin = model(inputs)
                    preds = preds.clamp(0.0,1.0)
                    
                    gaussian_blur = transforms.GaussianBlur((3,3),3)
                    labels_bin = labels.clone().detach()
                    masks_bin = masks.clone().detach()
                    masks_bin = F.interpolate(masks_bin, size=64)
                    labels_bin = gaussian_blur(labels_bin)
                    labels_bin = labels_bin.cpu().numpy()
                    labels_bin = labels_bin>0.2
                    labels_bin = labels_bin.astype("float32")
                    labels_bin = torch.from_numpy(labels_bin).to(device)
                    
                    P_SR = model_bpnn(masks_bin,preds_bin)
                    P_HR = model_bpnn(masks_bin,labels_bin)
                    BVTV_SR = bvtv_loss(preds_bin,masks)
                    BVTV_HR = bvtv_loss(labels,masks)
                    P_SR = torch.cat((P_SR,BVTV_SR),dim=1).clamp(-1,1)
                    P_HR = torch.cat((P_HR,BVTV_HR),dim=1).clamp(-1,1)

                    L_SR = criterion(preds, labels)
                    L_BPNN = Lbpnn(P_SR,P_HR)
                    loss = L_SR + (0 * L_BPNN)

                    epoch_losses.update(loss.item())
                    bpnn_loss.update(L_BPNN.item())
                    
                    optimizer.zero_grad()
                    loss.mean().backward()
                     
                    optimizer.step()

                    with torch.no_grad():
                        psnr_train.update(calc_psnr(labels.cpu(),preds.clamp(0.0,1.0).cpu(),masks.cpu(),device="cpu").item())
                        ssim_train.update(ssim(x=labels.cpu(),y=preds.clamp(0.0,1.0).cpu(),data_range=1.,downsample=False,mask=masks.cpu(),device="cpu").item())
                    
                    t.set_postfix(loss='{:.9f}'.format(epoch_losses.avg),LossSR='{:.9f}'.format(L_SR.item()),bpnn='{:.3f}'.format(bpnn_loss.avg),psnr='{:.1f}'.format(psnr_train.avg),ssim='{:.1f}'.format(ssim_train.avg),alpha='{:.8f}'.format(args.alpha[trial]))
                    t.update(len(inputs))
                        
            tr_psnr.append(psnr_train.avg)
            tr_ssim.append(ssim_train.avg)
            tr_score.append(epoch_losses.avg)
            tr_bpnn.append(bpnn_loss.avg)

            print("##### Train #####")
            print("Alpha =", args.alpha[trial])
            print("BPNN loss: {:.6f}".format(bpnn_loss.avg))
            print("train loss : {:.6f}".format(epoch_losses.avg))
            
            psnr_test = AverageMeter()
            ssim_test = AverageMeter()
            epoch_losses_test = AverageMeter()
            bpnn_loss_test = AverageMeter()
            for i,data in enumerate(test_dataloader):
                inputs, labels, masks, imagename = data
                inputs = inputs.reshape(inputs.size(0),1,256,256)
                labels = labels.reshape(labels.size(0),1,512,512)
                masks = masks.reshape(masks.size(0),1,512,512)

                inputs = inputs.to(device)
                labels = labels.to(device)
                masks = masks.to(device)
                with torch.no_grad():
                    preds,preds_bin = model(inputs)
                    preds = preds.clamp(0.0,1.0)
                    
                    gaussian_blur = transforms.GaussianBlur((3,3),3)
                    labels_bin = labels.clone().detach()
                    labels_bin = gaussian_blur(labels_bin)
                    labels_bin = labels_bin.cpu().numpy()
                    masks_bin = masks.clone().detach()
                    masks_bin = F.interpolate(masks_bin, size=64)
                    #t1, t2 = threshold_otsu(preds),threshold_otsu(labels)
                    labels_bin = labels_bin>0.2
                    labels_bin = labels_bin.astype("float32")
                    #preds_bin = preds_bin.astype("float32")
                    #preds_bin = torch.from_numpy(preds_bin).to(device)
                    labels_bin = torch.from_numpy(labels_bin).to(device)
                    P_SR = model_bpnn(masks_bin,preds_bin)
                    P_HR = model_bpnn(masks_bin,labels_bin)
                    BVTV_SR = bvtv_loss(preds_bin,masks)
                    #print("bvtv on SR:",BVTV_SR)
                    BVTV_HR = bvtv_loss(labels_bin,masks)
                    #print("bvtv on HR:",BVTV_HR)
                    P_SR = torch.cat((P_SR,BVTV_SR),dim=1).clamp(-1,1)
                    P_HR = torch.cat((P_HR,BVTV_HR),dim=1).clamp(-1,1)
                    Ltest_SR = criterion(preds, labels)
                    Ltest_BPNN = Lbpnn(P_SR,P_HR)
                    loss_test = Ltest_SR + (1e-4 * Ltest_BPNN)
                    #if best_epoch_tracking > loss_test.item():
                        #best_epoch_tracking = loss_test.item();
                        #if i%100:
                            #print("-")
                            #torchvision.utils.save_image(labels_bin, './save_image/alpha_'+str(args.alpha[trial]) + '/labels_bin_'+imagename[0])
                            #torchvision.utils.save_image(labels,'./save_image/alpha_'+str(args.alpha[trial])+'/labels_'+imagename[0])
                            #torchvision.utils.save_image(preds_bin,'./save_image/alpha_'+str(args.alpha[trial])+'/preds_bin_'+imagename[0])
                            #torchvision.utils.save_image(preds,'./save_image/alpha_'+str(args.alpha[trial])+'/preds'+imagename[0])
                
                    epoch_losses_test.update(loss_test.item())
                    bpnn_loss_test.update(Ltest_BPNN.item())
                    psnr_test.update(calc_psnr(labels.cpu(),preds.clamp(0.0,1.0).cpu(),masks.cpu(),device="cpu").item())
                    ssim_test.update(ssim(x=labels.cpu(),y=preds.clamp(0.0,1.0).cpu(),data_range=1.,downsample=False,mask=masks.cpu(),device="cpu").item())
            print("##### TEST #####")
            print("ssim",epoch_losses_test.avg)
            print("psnr", psnr_test.avg)
            t_score.append(epoch_losses_test.avg)
            t_bpnn.append(bpnn_loss_test.avg)
            t_psnr.append(psnr_test.avg)
            t_ssim.append(ssim_test.avg)
          #  if epoch_losses_test.avg < best_loss:
          #      best_epoch = epoch
          #      best_loss = epoch_losses_test.avg
                #best_weights = copy.deepcopy(model.state_dict())
    training_info = {"loss_train": tr_score,
                     "loss_test": t_score,
                     "bpnn_train" : tr_bpnn,
                     "bpnn_test": t_bpnn,
                     "psnr_test": t_psnr,
                     "ssim_test": t_ssim,
                     "train_ssim": tr_ssim,
                     "train_psnr": tr_psnr,
                     "alpha": args.alpha[trial]
                    }
    print(" ------------ CROSS RESULTS -------------")
    print("loss train:",training_info["loss_train"])
    print("loss test:",training_info['loss_test'])
    print("bpnn train:",training_info["bpnn_train"])
    print("alpha:",training_info["alpha"])
    i=1
    while os.path.exists(os.path.join(args.outputs_dir,"losses_info"+str(i)+".pkl")) == True:
        i=i+1
    with open( os.path.join(args.outputs_dir,"losses_info"+str(i)+".pkl"), "wb") as f:
        pickle.dump(training_info,f)
    return np.min(np.array(t_bpnn)),args.alpha[trial] 


study= {"loss":[],"alpha":[]}
for n_trial in range(6):
    lo,al = objective(n_trial)
    study["loss"].append(lo)
    study["alpha"].append(al)

    with open("Kbar_limit.pkl","wb") as f:
        pickle.dump(study,f)
