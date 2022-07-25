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
import pytorch_ssim
from tqdm import tqdm
import optuna
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from models import FSRCNN, BPNN
from datasets import TrainDataset
from utils import AverageMeter, metrics
import time

NB_DATA = 4474


def objective(trial):
    parser = argparse.ArgumentParser()
    parser.add_argument('--HR_dir', type=str,default = "../BPNN/data/ROI_trab/train")
    parser.add_argument('--LR_dir', type=str,default = "../BPNN/data/LR_trab/train")
    parser.add_argument('--mask_dir', type=str,default = "../BPNN/data/mask_trab/train")
    parser.add_argument('--outputs-dir', type=str, default = "./FSRCNN_search")
    parser.add_argument('--checkpoint_bpnn', type= str, default = "BPNN_checkpoint_75.pth")
    parser.add_argument('--alpha', default = [0,5*10**(-5),10**(-4),2.5*10**(-4),5*10**(-4),10**(-3),5*10**(-3),4*10**(-2),10**(-1),1])
    parser.add_argument('--Loss_bpnn', default = MSELoss)
    parser.add_argument('--weights-file', type=str)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=150)
    parser.add_argument('--num-workers', type=int, default=6)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--nof', type= int, default = 85)
    parser.add_argument('--n1', type=int,default = 158)
    parser.add_argument('--n2', type=int,default = 211)
    parser.add_argument('--n3', type=int,default = 176)
    parser.add_argument('--gpu_ids', type=list, default = [0])
    parser.add_argument('--NB_LABEL', type=int, default = 6)
    parser.add_argument('--k_fold', type=int, default = 1)
    args = parser.parse_args()

    args.outputs_dir = os.path.join(args.outputs_dir, 'BPNN_psnr_mask_x{}'.format(args.scale))
    
    if os.path.exists(args.outputs_dir) == False:
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count())
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model_bpnn = BPNN(args.nof, args.NB_LABEL, n1= args.n1, n2=args.n2, n3=args.n3, k1=3,k2=3,k3=3).to(device)
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
        kf = train_test_split(index,test_size=0.2,random_state=42)
    cross_bpnn, cross_score, cross_psnr = [], [], []
    for k in range(args.k_fold):
        train_index = kf[0]
        test_index = kf[1]
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

        
        dataset = TrainDataset(args.HR_dir, args.LR_dir)
        train_dataloader = DataLoader(dataset=dataset,
                                      batch_size=args.batch_size,
                                      sampler=train_index,
                                      num_workers=args.num_workers)
        eval_dataloader = DataLoader(dataset=dataset, 
                                     sampler=test_index,
                                     batch_size=1,
                                     num_workers=args.num_workers)    

        best_weights = copy.deepcopy(model.state_dict())
        best_epoch = 0
        best_loss = 10
        t_score, tr_score, tr_bpnn, t_bpnn, t_psnr,t_ssim = [], [] ,[], [], [], []
        start = time.time()
        cross_bpnn, cross_score, cross_psnr, cross_ssim = np.zeros(args.num_epochs),np.zeros(args.num_epochs),np.zeros(args.num_epochs),np.zeros(args.num_epochs)
        
        for epoch in range(args.num_epochs):
            model.train()
            epoch_losses = AverageMeter()
            bpnn_loss = AverageMeter()

            with tqdm(total=(len(train_dataloader) - len(train_dataloader) % args.batch_size), ncols=80) as t:
                t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

                for data in train_dataloader:
                    inputs, labels, _ = data
                    inputs = inputs.reshape(inputs.size(0),1,256,256)
                    labels = labels.reshape(labels.size(0),1,512,512)
                    inputs, labels= inputs.float(), labels.float()
                    inputs, labels = inputs.to(device), labels.to(device)
                    preds = model(inputs)
                    P_SR = model_bpnn(preds)
                    P_HR = model_bpnn(labels)

                    L_SR = criterion(preds, labels)
                    L_BPNN = Lbpnn(P_SR,P_HR)
                    loss = L_SR + (args.alpha[trial] * L_BPNN)

                    epoch_losses.update(loss.item(), len(inputs))
                    bpnn_loss.update(L_BPNN.item(), len(inputs))
                    optimizer.zero_grad()
                    #loss.backward()
                    loss.mean().backward()
                    optimizer.step()

                    t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                    t.update(len(inputs))
            print("##### Train #####")
            print("BPNN loss: {:.6f}".format(bpnn_loss.avg))
            print("train loss : {:.6f}".format(epoch_losses.avg))
            #torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))
            print("epoch_losses type :", type(epoch_losses.avg))
            print("bpnn_loss", type(bpnn_loss.avg))
            tr_score.append(epoch_losses.avg)
            tr_bpnn.append(bpnn_loss.avg)
            psnr = []
            ssim_list = []
            model.eval()
            epoch_losses_test = AverageMeter()
            bpnn_loss_test = AverageMeter()
            for data in eval_dataloader:
                inputs, labels, imagename = data
                inputs = inputs.reshape(inputs.size(0),1,256,256)
                labels = labels.reshape(labels.size(0),1,512,512)
                inputs, labels = inputs.float(), labels.float()
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    preds = model(inputs).clamp(0.0, 1.0)
                    Ltest_SR = criterion(preds, labels)
                    Ltest_BPNN = Lbpnn(P_SR,P_HR)
                    loss_test = Ltest_SR + (args.alpha[trial] * Ltest_BPNN)
                    epoch_losses_test.update(loss_test.item())
                    bpnn_loss_test.update(Ltest_BPNN.item())
                    psnr_ssim = utils.metrics(labels,preds,args.mask_dir,imagename)
                    psnr.append(psnr_ssim.calc_psnr.item())
                    ssim_list.append(psnr_ssim.calc_ssim.item())
            print("##### Test #####")
            print('eval loss: {:.6f}'.format(epoch_losses_test.avg))
            print('bpnn loss: {:.6f}'.format(bpnn_loss_test.avg))
            t_score.append(epoch_losses_test.avg)
            t_bpnn.append(bpnn_loss_test.avg)
            t_psnr.append(sum(psnr)/len(psnr))
            t_ssim.append(sum(ssim_list)/len(ssim_list))
            if epoch_losses_test.avg < best_loss:
                best_epoch = epoch
                best_loss = epoch_losses_test.avg
                #best_weights = copy.deepcopy(model.state_dict())
        end = time.time() 
        print("Time :", end-start) 
        cross_bpnn = cross_bpnn + np.array(t_bpnn)
        cross_score = cross_score + np.array(t_score)
        cross_psnr = cross_psnr + np.array(t_psnr)
        cross_ssim = cross_ssim + np.array(t_ssim)
    print("bpnn :",cross_bpnn/args.k_fold)
    print("score :", cross_score/args.k_fold)
    print("psnr :", cross_psnr/args.k_fold)
    print("ssim :", cross_ssim/args.k_fold)
    print("tr_bpnn:", tr_bpnn)
    training_info = {"loss_train": tr_score, "loss_val": cross_score/args.k_fold, "bpnn_train" : tr_bpnn, "bpnn_val": cross_bpnn/args.k_fold, "psnr": cross_psnr/args.k_fold, "ssim":cross_ssim/args.k_fold}
    i=1
    while os.path.exists(os.path.join(args.outputs_dir,"losses_info"+str(i)+".pkl")) == True:
        i=i+1
    with open( os.path.join(args.outputs_dir,"losses_info"+str(i)+".pkl"), "wb") as f:
        pickle.dump(training_info,f)
    print('best epoch: {}, loss: {:.6f}'.format(best_epoch, best_loss))
    return np.min(cross_bpnn/args.k_fold), np.max(t_psnr/args.k_fold), args.alpha[trial], np.max(cross_ssim/args.k_fold)
        #torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))

study= {"bpnn" :[], "psnr": [], "alpha": [],"ssim":[]}
for n_trial in range(8):
    bp,ps,al,ss = objective(n_trial)
    study["bpnn"].append(bp)
    study["psnr"].append(ps)
    study["alpha"].append(al)
    study["ssim"].append(ss)

with open("./FSRCNN_mask.pkl","wb") as f:
    pickle.dump(study,f)
