import argparse
import os
import copy
import pickle
import torch
from torch import nn
from torch.nn import L1Loss
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import optuna
import joblib
from sklearn.model_selection import train_test_split
from models import FSRCNN, BPNN
from datasets import TrainDataset
from utils import AverageMeter, calc_psnr
import time

def objective(trial):
    parser = argparse.ArgumentParser()
    parser.add_argument('--HR_dir', type=str,default = "../BPNN/data/ROI_trab")
    parser.add_argument('--LR_dir', type=str,default = "../BPNN/data/LR_trab")
    parser.add_argument('--outputs-dir', type=str, default = "./FSRCNN_search")
    parser.add_argument('--checkpoint_bpnn', type= str, default = "BPNN_checkpoint_51.pth")
    parser.add_argument('--alpha', default = trial.suggest_loguniform("alpha",1e-5,1e6))
    parser.add_argument('--weights-file', type=str)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--nof', type= int, default = 19)
    parser.add_argument('--n1', type=int,default = 179)
    parser.add_argument('--n2', type=int,default = 182)
    parser.add_argument('--n3', type=int,default = 190)
    parser.add_argument('--gpu_ids', type=list, default = [0,1])
    parser.add_argument('--NB_LABEL', type=int, default = 5)
    args = parser.parse_args()

    args.outputs_dir = os.path.join(args.outputs_dir, 'BPNN_first_shot_x{}'.format(args.scale))
    
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)
    print('how many times do we start again ?')
    model = FSRCNN(scale_factor=args.scale).to(device)
    model_bpnn = BPNN(args.nof, args.NB_LABEL, n1= args.n1, n2=args.n2, n3=args.n3, k1=3,k2=3,k3=3).to(device)
    model_bpnn.load_state_dict(torch.load(os.path.join(args.checkpoint_bpnn)))

    criterion = nn.MSELoss()
    Lbpnn = L1Loss()
    optimizer = optim.Adam([
        {'params': model.first_part.parameters()},
        {'params': model.mid_part.parameters()},
        {'params': model.last_part.parameters(), 'lr': args.lr * 0.1}
    ], lr=args.lr)
    
    

    train_dataset = TrainDataset(args.HR_dir,args.LR_dir)
    [trainset,valset] = train_test_split(train_dataset,test_size = 0.2, random_state=1)
    train_dataloader = DataLoader(dataset=trainset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True)
    eval_dataloader = DataLoader(dataset=valset, batch_size=1)    

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_loss = 10
    t_score = []
    tr_score = []
    tr_bpnn = []
    t_bpnn = []
    start = time.time()
    for epoch in range(args.num_epochs):
        model.train()
        epoch_losses = AverageMeter()
        bpnn_loss = AverageMeter()

        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size), ncols=80) as t:
            t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

            for data in train_dataloader:
                inputs, labels = data
                inputs = inputs.reshape(inputs.size(0),1,256,256)
                labels = labels.reshape(labels.size(0),1,512,512)
                inputs, labels= inputs.float(), labels.float()
                inputs = inputs.to(device)
                labels = labels.to(device)
                preds = model(inputs)
                P_SR = model_bpnn(preds)
                P_HR = model_bpnn(labels)
                
                L_SR = criterion(preds, labels)
                L_BPNN = Lbpnn(P_HR,P_SR)
                loss = L_SR + (args.alpha * L_BPNN)
                
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
        tr_score.append(epoch_losses.avg)
        tr_bpnn.append(bpnn_loss.avg)
        
        model.eval()
        epoch_losses_test = AverageMeter()
        bpnn_loss_test = AverageMeter()
        for data in eval_dataloader:
            inputs, labels = data
            
            inputs = inputs.reshape(inputs.size(0),1,256,256)
            labels = labels.reshape(labels.size(0),1,512,512)
            inputs, labels = inputs.float(), labels.float()
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = model(inputs).clamp(0.0, 1.0)
                Ltest_SR = criterion(preds, labels)
                Ltest_BPNN = Lbpnn(P_HR,P_SR)
                loss_test = Ltest_SR + (args.alpha * Ltest_BPNN)
                epoch_losses_test.update(loss_test.item())
                bpnn_loss_test.update(Ltest_BPNN.item())
        print("##### Test #####")
        print('eval loss: {:.6f}'.format(epoch_losses_test.avg))
        print('bpnn loss: {:.6f}'.format(bpnn_loss_test.avg))
        t_score.append(epoch_losses_test.avg)
        t_bpnn.append(bpnn_loss_test.avg)
        if epoch_losses_test.avg < best_loss:
            best_epoch = epoch
            best_loss = epoch_losses_test.avg
            #best_weights = copy.deepcopy(model.state_dict())
    end = time.time() 
    print("Time :", end-start) 
    training_info = {"loss_train": tr_score, "loss_test": t_score, "bpnn_train" : tr_bpnn, "bpnn_test": t_bpnn}
    with open( os.path.join(args.outputs_dir,"losses_info.pkl"), "wb") as f:
        pickle.dump(training_info,f)
    print('best epoch: {}, loss: {:.6f}'.format(best_epoch, best_loss))
    return best_loss
    #torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))
    
study = optuna.create_study(sampler=optuna.samplers.TPESampler(), direction='minimize')
study.optimize(objective,n_trials=12)
with open("./FSRCNN_BPNN_search.pkl","wb") as f:
    pickle.dump(study,f)
