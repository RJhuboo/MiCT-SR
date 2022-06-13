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
import optuna
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from models import FSRCNN, BPNN
from datasets import TrainDataset
from utils import AverageMeter, calc_psnr

NB_DATA = 4474

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--HR_dir', type=str,default = "../BPNN/data/ROI_trab/train")
    parser.add_argument('--LR_dir', type=str,default = "../BPNN/data/LR_trab/train")
    parser.add_argument('--outputs-dir', type=str, default = "./FSRCNN_search")
    parser.add_argument('--checkpoint_bpnn', type= str, default = "BPNN_checkpoint_75.pth")
    parser.add_argument('--alpha',type=float, default = 1e-5)
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
    parser.add_argument('--k_fold', type=int, default = 5)
    args = parser.parse_args()

    args.outputs_dir = os.path.join(args.outputs_dir, 'BPNN_training_x{}'.format(args.scale))
    
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
    model_bpnn = BPNN(args.nof, args.NB_LABEL, n1= args.n1, n2=args.n2, n3=args.n3, k1=3,k2=3,k3=3).to(device)
    model_bpnn.load_state_dict(torch.load(os.path.join(args.checkpoint_bpnn)))
    if torch.cuda.device_count() > 1:
        model_bpnn = nn.DataParallel(model_bpnn)
    model_bpnn.to(device)
    
    # Freeze BPNN
    for param in model_bpnn.parameters():
        param.requires_grad = False
    model_bpnn.eval()
                       
    cross_bpnn, cross_score, cross_psnr = [], [], []
    
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
    tr_score, tr_bpnn= [], []

    # Start epoch
    for epoch in range(args.num_epochs):
      
        model.train()
        epoch_losses = AverageMeter()
        bpnn_loss = AverageMeter()
        
        with tqdm(total=(len(train_dataloader) - len(train_dataloader) % args.batch_size), ncols=80) as t:
            t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))
            
            # Start training
            for data in train_dataloader:
                
                # Forward
                inputs, labels = data
                inputs = inputs.reshape(inputs.size(0),1,256,256)
                labels = labels.reshape(labels.size(0),1,512,512)
                inputs, labels= inputs.float(), labels.float()
                inputs, labels = inputs.to(device), labels.to(device)
                preds = model(inputs)
                P_SR = model_bpnn(preds)
                P_HR = model_bpnn(labels)

                # Loss
                L_SR = criterion(preds, labels)
                L_BPNN = Lbpnn(P_SR,P_HR)
                print("LSR:",L_SR)
                print("LBPNN:",L_BPNN)
                loss = L_SR + (args.alpha * L_BPNN)

                epoch_losses.update(loss.item(), len(inputs))
                bpnn_loss.update(L_BPNN.item(), len(inputs))
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                #loss.mean().backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(inputs))
        
        # Display result Losses
        print("##### Train #####")
        print("BPNN loss: {:.6f}".format(bpnn_loss.avg))
        print("train loss : {:.6f}".format(epoch_losses.avg))
        
        # Save model
        torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))

        tr_score.append(epoch_losses.avg)
        tr_bpnn.append(bpnn_loss.avg)
    
    # Save training losses
    training_info = {"loss_train": tr_score,"bpnn_train" : tr_bpnn}
    i=1
    while os.path.exists(os.path.join(args.outputs_dir,"losses_info"+str(i)+".pkl")) == True:
        i=i+1
    with open( os.path.join(args.outputs_dir,"losses_info"+str(i)+".pkl"), "wb") as f:
        pickle.dump(training_info,f)
    print('best epoch: {}, loss: {:.6f}'.format(best_epoch, best_loss))
        #torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))

