from torch.autograd import Variable
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
import torchvision.transforms as transforms
import pytorch_ssim
from tqdm import tqdm
import optuna
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from models import FSRCNN, BPNN
from datasets import TrainDataset
from utils import AverageMeter, calc_psnr
from ssim import ssim
import time

NB_DATA = 7100


def objective(trial):
    parser = argparse.ArgumentParser()
    parser.add_argument('--HR_dir', type=str,default = "./data/HR/Train_Label_trab_100")
    parser.add_argument('--LR_dir', type=str,default = "./data/LR/Train_trab")
    parser.add_argument('--mask_dir',type=str,default = "./data/HR/Train_trab_mask")
    parser.add_argument('--outputs-dir', type=str, default = "./FSRCNN_search")
    parser.add_argument('--checkpoint_bpnn', type= str, default = "./checkpoints_bpnn/BPNN_checkpoint_149.pth")
    parser.add_argument('--alpha', default = [0,10**(-3),10**(-1),10**(-4),5*10**(-4),5*10**(-3),4*10**(-2),5*10**(-5)])
    parser.add_argument('--Loss_bpnn', default = MSELoss)
    parser.add_argument('--weights-file', type=str)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--num-workers', type=int, default=6)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--nof', type= int, default = 36)
    parser.add_argument('--n1', type=int,default = 135)
    parser.add_argument('--n2', type=int,default = 146)
    parser.add_argument('--n3', type=int,default = 131)
    parser.add_argument('--gpu_ids', type=list, default = [0, 1, 3])
    parser.add_argument('--NB_LABEL', type=int, default = 9)
    parser.add_argument('--k_fold', type=int, default = 1)
    args = parser.parse_args()

    args.outputs_dir = os.path.join(args.outputs_dir, 'BPNN_9p_x{}'.format(args.scale))
    
    if os.path.exists(args.outputs_dir) == False:
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count())
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
           
    #train_dataset = TrainDataset(args.HR_dir,args.LR_dir)
    index = range(NB_DATA)
    if args.k_fold >1:
        kf = KFold(n_splits = args.k_fold, shuffle=True)
    else:
        kf = train_test_split(index,train_size=6000,test_size=1100,random_state=42)
    cross_bpnn, cross_score, cross_psnr, cross_ssim = np.zeros(args.num_epochs),np.zeros(args.num_epochs),np.zeros(args.num_epochs),np.zeros(args.num_epochs)
    cross_bpnn_train, cross_score_train, cross_psnr_train, cross_ssim_train = np.zeros(args.num_epochs),np.zeros(args.num_epochs),np.zeros(args.num_epochs),np.zeros(args.num_epochs)
    for k in range(1):
    # for train_index, test_index in kf.split(index):
        train_index = kf[0]
        test_index = kf[1]
        print("-------  Data separation -------")
        print("train size:",len(train_index))
        print("test size:",len(test_index))
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
        criterion = nn.MSELoss()
        Lbpnn =  args.Loss_bpnn()
        
        my_transforms = transforms.Compose([
          transforms.ToPILImage(),
          transforms.RandomRotation(degrees=45),
          transforms.RandomHorizontalFlip(p=0.3),
          transforms.RandomVerticalFlip(p=0.3),
          transforms.RandomAffine(degrees=(0,1),translate=(0.1,0.1)),
          transforms.ToTensor(),
        ])
        
        dataset = TrainDataset(args.HR_dir, args.LR_dir, args.mask_dir,transform = my_transforms)
        train_dataloader = DataLoader(dataset=dataset,
                                      batch_size=args.batch_size,
                                      sampler=train_index,
                                      num_workers=args.num_workers)
        print(len(train_dataloader))
        eval_dataloader = DataLoader(dataset=dataset, 
                                     sampler=test_index,
                                     batch_size=1,
                                     num_workers=args.num_workers)    
        print(len(eval_dataloader))
        best_weights = copy.deepcopy(model.state_dict())
        best_epoch = 0
        best_loss = 10
        tr_psnr = []
        tr_ssim = []
        t_score, tr_score, tr_bpnn, t_bpnn, t_psnr,t_ssim = [], [] ,[], [], [], []
        start = time.time()

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
                    #inputs, labels, masks = inputs.float(), labels.float(), masks.float()
                    inputs, labels, masks = inputs.to(device), labels.to(device), masks.to(device)
                    
                    preds = model(inputs)
                    
                    model_bpnn = BPNN(in_channel=1,features=args.nof, out_channels=args.NB_LABEL, n1= args.n1, n2=args.n2, n3=args.n3, k1=3,k2=3,k3=3).to(device)
                    model_bpnn.load_state_dict(torch.load(os.path.join(args.checkpoint_bpnn)))
                    if torch.cuda.device_count() > 1:
                        model_bpnn = nn.DataParallel(model_bpnn)
                    model_bpnn.to(device)
                    for param in model_bpnn.parameters():
                        param.requires_grad = False
                    model_bpnn.eval()
    
                    P_SR = model_bpnn(masks,preds)
                    P_HR = model_bpnn(masks,labels)
                    del model_bpnn
                
                    L_SR = criterion(preds, labels)
                    L_BPNN = Lbpnn(P_SR,P_HR)
                                        
                    loss = L_SR + (args.alpha[trial] * L_BPNN)
                    
                    epoch_losses.update(loss.item(),len(inputs))
                    bpnn_loss.update(L_BPNN.item(),len(inputs))
                    optimizer.zero_grad()
                    #loss.backward()
                    loss.mean().backward()
                    optimizer.step()

                    t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                    t.update(len(inputs))
                    #preds = model(inputs).clamp(0.0, 1.0)
                    psnr_train.update(calc_psnr(labels.cpu(),preds.clamp(0.0,1.0).cpu(),args.mask_dir,imagename,device="cpu").item())
                    ssim_train.update(ssim(x=labels.cpu(),y=preds.clamp(0.0,1.0).cpu(),data_range=1.,downsample=False,directory = args.mask_dir,maskname = imagename,device="cpu"))
            
            tr_psnr.append(psnr_train.avg)
            tr_ssim.append(ssim_train.avg)
            tr_score.append(epoch_losses.avg)
            tr_bpnn.append(bpnn_loss.avg)
            print("##### Train #####")
            print("BPNN loss: {:.6f}".format(bpnn_loss.avg))
            print("train loss : {:.6f}".format(epoch_losses.avg))
            #torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))
            del psnr_train,ssim_train,epoch_losses,bpnn_loss
            
            psnr = AverageMeter()
            ssim_list = AverageMeter()
            model.eval()
            epoch_losses_test = AverageMeter()
            bpnn_loss_test = AverageMeter()
            for data in eval_dataloader:
                inputs, labels, masks, imagename = data
                inputs = inputs.reshape(inputs.size(0),1,256,256)
                labels = labels.reshape(labels.size(0),1,512,512)
                masks = masks.reshape(masks.size(0),1,512,512)
                #inputs, labels, masks = inputs.float(), labels.float(), masks.float()
                
                inputs = inputs.to(device)
                labels = labels.to(device)
                masks = masks.to(device)
                with torch.no_grad():
                    preds = model(inputs).clamp(0.0,1.0)
                    model_bpnn = BPNN(in_channel=1,features=args.nof, out_channels=args.NB_LABEL, n1= args.n1, n2=args.n2, n3=args.n3, k1=3,k2=3,k3=3).to(device)
                    model_bpnn.load_state_dict(torch.load(os.path.join(args.checkpoint_bpnn)))
                    if torch.cuda.device_count() > 1:
                        model_bpnn = nn.DataParallel(model_bpnn)
                    model_bpnn.to(device)
                    for param in model_bpnn.parameters():
                        param.requires_grad = False
                    model_bpnn.eval()
                    P_SR = model_bpnn(masks,preds)
                    P_HR = model_bpnn(masks,labels)
                    del model_bpnn
                    Ltest_SR = criterion(preds, labels) 
                    Ltest_BPNN = Lbpnn(P_SR,P_HR)
                    loss_test = Ltest_SR + (args.alpha[trial] * Ltest_BPNN)
                    epoch_losses_test.update(loss_test.item())
                    bpnn_loss_test.update(Ltest_BPNN.item())
                    psnr.update(calc_psnr(labels,preds,args.mask_dir,imagename,device).item())
                    ssim_list.update(ssim(x=labels,y=preds,data_range=1.,downsample=False,directory = args.mask_dir,maskname = imagename))
            print("##### Test #####")
            print('eval loss: {:.6f}'.format(epoch_losses_test.avg))
            print('bpnn loss: {:.6f}'.format(bpnn_loss_test.avg))
            print('psnr : {:.6f}'.format(psnr.avg))
            print('ssim : {:.6f}'.format(ssim_list.avg))
            t_score.append(epoch_losses_test.avg)
            t_bpnn.append(bpnn_loss_test.avg)
            t_psnr.append(psnr.avg)
            t_ssim.append(ssim_list.avg)
            if epoch_losses_test.avg < best_loss:
                best_epoch = epoch
                best_loss = epoch_losses_test.avg
                #best_weights = copy.deepcopy(model.state_dict())
            del epoch_losses_test, bpnn_loss_test, psnr, ssim_list
        end = time.time() 
        print("Time :", end-start) 
        cross_bpnn = cross_bpnn + np.array(t_bpnn)
        cross_score = cross_score + np.array(t_score)
        cross_psnr = cross_psnr + np.array(t_psnr)
        cross_ssim = cross_ssim + np.array(t_ssim)
        cross_bpnn_train = cross_bpnn_train +np.array(tr_bpnn)
        cross_score_train = cross_score_train + np.array(tr_score)
        cross_psnr_train = cross_psnr_train +np.array(tr_psnr)
        cross_ssim_train = cross_ssim_train + np.array(ssim_psnr)
        del tr_psnr, tr_ssim, tr_score,tr_bpnn, t_bpnn,t_score,t_psnr,t_ssim
                  
    training_info = {"loss_train": cross_psnr_train/args.k_fold,
                     "loss_val": cross_score/args.k_fold,
                     "bpnn_train" :cross_bpnn_train/args.k_fold,
                     "bpnn_val": cross_bpnn/args.k_fold,
                     "psnr": cross_psnr/args.k_fold,
                     "ssim":cross_ssim/args.k_fold,
                     "train_ssim": cross_ssim_train/args.k_fold,
                     "train_psnr": cross_psnr_train/args.k_fold
                    }
    i=1
    while os.path.exists(os.path.join(args.outputs_dir,"losses_info"+str(i)+".pkl")) == True:
        i=i+1
    with open( os.path.join(args.outputs_dir,"losses_info"+str(i)+".pkl"), "wb") as f:
        pickle.dump(training_info,f)
    print('best epoch: {}, loss: {:.6f}'.format(best_epoch, best_loss))
    return np.min(np.array(cross_bpnn)/args.k_fold), np.max(np.array(cross_psnr)/args.k_fold), args.alpha[trial], np.max(np.array(cross_ssim)/args.k_fold)
        #torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))

study= {"bpnn" :[], "psnr": [], "alpha": [],"ssim":[]}
for n_trial in range(8):
    bp,ps,al,ss = objective(n_trial)
    study["bpnn"].append(bp)
    study["psnr"].append(ps)
    study["alpha"].append(al)
    study["ssim"].append(ss)

with open("./FSRCNN_9p.pkl","wb") as f:
    pickle.dump(study,f)
