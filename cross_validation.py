fom torch.autograd import Variable
import argparse
import os
import copy
import pickle
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import MSELoss
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms
import torchvision
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from models import FSRCNN, BPNN
from datasets import TrainDataset, TestDataset
from utils import AverageMeter, calc_psnr, MorphLoss
from ssim import ssim
import time
from PIL import Image
import pandas as pd

def normalization(csv_file="./Label_trab_FSRCNN.csv",mode="standardization",indices=range(5800)):
    Data = pd.read_csv(csv_file)
    if mode == "standardization":
        scaler = StandardScaler()
    scaler.fit(Data.iloc[indices,1:])
    return scaler

NB_DATA = 7100
def bvtv_loss(tensor_to_count,tensor_mask):
    tensor_to_count = (tensor_to_count>0.2).type(torch.float32)*1.
    ones = (tensor_to_count == 1).sum(dim=2)
    BV = ones.sum(dim=2)
    ones = (tensor_mask == 1).sum(dim=2)
    TV = ones.sum(dim=2)
    BVTV = BV/TV
    return BVTV

def MSE(y_predicted,y,batch_size):
    squared_error = abs((y_predicted.cpu().detach().numpy() - y.cpu().detach().numpy()))**2
    return squared_error

def objective(trial):
    parser = argparse.ArgumentParser()
    parser.add_argument('--HR_dir', type=str,default = "/gpfsstore/rech/tvs/uki75tv/data_fsrcnn/HR/Train_Label_trab_100")
    parser.add_argument('--LR_dir', type=str,default = "/gpfsstore/rech/tvs/uki75tv/data_fsrcnn/LR/Train_trab")
    parser.add_argument('--mask_dir',type=str,default = "/gpfsstore/rech/tvs/uki75tv/data_fsrcnn/HR/Train_trab_mask")
    parser.add_argument('--tensorboard_name',type=str,default = "BPNN_x2")
    parser.add_argument('--outputs-dir', type=str, default = "./FSRCNN_search")
    parser.add_argument('--checkpoint_bpnn', type= str, default =  "../BPNN/convnet_fsrcnn_adapted/BPNN_checkpoint_12.pth")
    parser.add_argument('--alpha', type = list, default = [1e-4,0])
    parser.add_argument('--Loss_bpnn', default = MSELoss)
    parser.add_argument('--weights-file', type=str)
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5)#-2
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
    parser.add_argument('--name', type=str, default = "BPNN_x4_last")
    args = parser.parse_args()
    
    ## Create summary for tensorboard
    writer = SummaryWriter(log_dir='runs/'+args.tensorboard_name)

    # Create output folder
    args.outputs_dir = os.path.join(args.outputs_dir, args.name)    
    if os.path.exists(args.outputs_dir) == False:
        os.makedirs(args.outputs_dir)
    if os.path.exists(args.outputs_dir + "/alpha_"+str(args.alpha[trial])) == False:
        os.makedirs(args.outputs_dir + "/alpha_"+str(args.alpha[trial]))

    # GPU
    cudnn.benchmark = True
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count())
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load MPNN and freeze the layer
    model_bpnn = BPNN(in_channel=1,features=args.nof, out_channels=args.NB_LABEL, n1= args.n1, n2=args.n2, n3=args.n3, k1=3,k2=3,k3=3).to(device)
    model_bpnn.load_state_dict(torch.load(os.path.join(args.checkpoint_bpnn)))
    if torch.cuda.device_count() > 1:
        model_bpnn = nn.DataParallel(model_bpnn)
    model_bpnn.to(device)
    for param in model_bpnn.parameters():
        param.requires_grad = False
    model_bpnn.eval()

    # Load parameters to evaluate the performance 
    eval_data = pd.read_csv("./Trab2D_eval.csv")

    # DATA SPLITTING
    if args.k_fold >1:
        kf = KFold(n_splits = args.k_fold, shuffle=True)
    else:
        # Data splitting (take full mouse)
        index = list(range(71))
        kf = train_test_split(index,train_size=60,test_size=11,shuffle=True,random_state=42)
        kf[0] = sorted(kf[0])
        new_kf = [list(range(kf[0][i]*100,(kf[0][i]+1)*100)) if kf[0][i] != 52 else [] for i in range(60)]
        new_kf = [sublist for sublist in new_kf if sublist]
        new_kf=np.array(new_kf)
        kf[0] = np.vstack(new_kf).reshape((-1,1)).flatten()
        new_kf = [list(range(kf[1][i]*100,(kf[1][i]+1)*100)) for i in range(11)]
        new_kf = np.array(new_kf)
        kf[1] = np.vstack(new_kf).reshape((1100,1)).flatten()

    # STARTING VALIDATION EXPERIMENT
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
        
        if torch.cuda.device_count() >1:
            model = nn.DataParallel(model) 
        model.to(device)
        
        criterion = nn.MSELoss()
        Lbpnn =  nn.MSELoss()
        
        scaler = normalization()

        dataset = TrainDataset(args.HR_dir, args.LR_dir, args.mask_dir)
        dataset_test = TestDataset("/gpfsstore/rech/tvs/uki75tv/Test_trab","/gpfsstore/rech/tvs/uki75tv/data_fsrcnn/LR/Test_trab","/gpfsstore/rech/tvs/uki75tv/Test_trab_mask")
        train_dataloader = DataLoader(dataset=dataset,
                                      batch_size=args.batch_size,
                                      sampler=train_index,
                                      num_workers=args.num_workers)
        eval_dataloader = DataLoader(dataset=dataset, 
                                     sampler=test_index,
                                     batch_size=1,
                                     num_workers=args.num_workers)
        test_dataloader = DataLoader(dataset=dataset_test,batch_size=1,num_workers=args.num_workers)
        
        best_weights = copy.deepcopy(model.state_dict())
        best_epoch = 0
        best_loss = 10
        e_score, e_bpnn, e_psnr,e_ssim = [], [], [], []
        for epoch in range(args.num_epochs):
            model.train()
            epoch_losses = AverageMeter()
            bpnn_loss = AverageMeter()
            psnr_train = AverageMeter()
            ssim_train = AverageMeter()
            with tqdm(total=(len(train_dataloader) - len(train_dataloader) % args.batch_size), ncols=80) as t:
                t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

                # First step: Training
                for data in train_dataloader:

                    inputs, labels, masks, imagename = data
                    
                    inputs = inputs.reshape(inputs.size(0),1,inputs.size(2),inputs.size(2))
                    labels = labels.reshape(labels.size(0),1,512,512)
                    labels_bin = labels_bin.reshape(labels_bin.size(0),1,512,512)
                    masks = masks.reshape(masks.size(0),1,512,512)
                    inputs, labels, masks = inputs.to(device), labels.to(device), masks.to(device)
                                        
                    preds,preds_bin = model(inputs)
                    preds = preds.clamp(0.0,1.0)
                    
                    labels_bin = labels.clone().detach()
                    masks_bin = masks.clone().detach()
        
                    masks_bin = F.interpolate(masks_bin, size=64)

                    # Binarized the label image
                    gaussian_blur = transforms.GaussianBlur((3,3),3)
                    labels_bin = gaussian_blur(labels_bin)
                    labels_bin = labels_bin.cpu().numpy()
                    labels_bin = labels_bin>0.225
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
                    loss = L_SR + (args.alpha[trial] * L_BPNN)

                    # Performance
                    epoch_losses.update(loss.item())
                    bpnn_loss.update(L_BPNN.item())
                    optimizer.zero_grad()
                    loss.mean().backward()
                    optimizer.step()
                    with torch.no_grad():
                        psnr_train.update(calc_psnr(labels.cpu(),preds.clamp(0.0,1.0).cpu(),masks.cpu(),device="cpu").item())
                        ssim_train.update(ssim(x=labels.cpu(),y=preds.clamp(0.0,1.0).cpu(),data_range=1.,downsample=False,mask=masks.cpu(),device="cpu").item())

                    # Display the performance
                    t.set_postfix(loss='{:.9f}'.format(epoch_losses.avg),LossSR='{:.9f}'.format(L_SR.item()),bpnn='{:.3f}'.format(bpnn_loss.avg),psnr='{:.1f}'.format(psnr_train.avg),ssim='{:.1f}'.format(ssim_train.avg),alpha='{:.8f}'.format(args.alpha[trial]))
                    t.update(len(inputs))

            # Training: Summary of the epoch
            print("##### Train #####")
            print("Alpha =", args.alpha[trial])
            print("BPNN loss: {:.6f}".format(bpnn_loss.avg))
            print("train loss : {:.6f}".format(epoch_losses.avg))
            
            psnr = AverageMeter()
            ssim_list = AverageMeter()
            epoch_losses_eval = AverageMeter()
            bpnn_loss_eval = AverageMeter()
            output_param= np.zeros((1100,8))
            label_param=np.zeros((1100,8))
            eval_param= np.zeros((1100,7))

            # Second step: Evaluation
            model.eval()
            for data in eval_dataloader:

                inputs, labels, masks, imagename = data
                inputs = inputs.reshape(inputs.size(0),1,inputs.size(2),inputs.size(2))
                labels = labels.reshape(labels.size(0),1,512,512)
                labels_bin = labels_bin.reshape(labels_bin.size(0),1,512,512)
                masks = masks.reshape(masks.size(0),1,512,512)
                
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels_bin = labels_bin.to(device)
                masks = masks.to(device)
                with torch.no_grad():
                    preds=model(inputs)
                    preds,preds_bin = model(inputs)
                    preds=preds.clamp(0.0,1.0)
                    masks_bin = masks.clone().detach()
                    masks_bin = F.interpolate(masks_bin, size=64)
                    
                    gaussian_blur = transforms.GaussianBlur((3,3),3)
                    labels_bin = gaussian_blur(labels_bin)
                    labels_bin = labels_bin>0.225
                    labels_bin = labels_bin.astype("float32")

                    P_SR = model_bpnn(masks_bin,preds_bin)
                    P_HR = model_bpnn(masks_bin,labels_bin)
                    
                    BVTV_SR = bvtv_loss(preds_bin,masks)
                    BVTV_HR = bvtv_loss(labels_bin,masks)
                    
                    P_SR = torch.cat((P_SR,BVTV_SR),dim=1)
                    P_HR = torch.cat((P_HR,BVTV_HR),dim=1
                    Leval_SR = criterion(preds*torch.Tensor(masks_bin), labels)
                    Leval_BPNN = Lbpnn(P_SR,P_HR)

                    loss_eval = Leval_SR + (args.alpha[trial] * Leval_BPNN)
                    epoch_losses_eval.update(loss_eval.item())
                    bpnn_loss_eval.update(Leval_BPNN.item())
                    psnr.update(calc_psnr(labels.cpu(),preds.clamp(0,1).cpu(),masks.cpu(),device="cpu").item())
                    ssim_list.update(ssim(x=labels.cpu(),y=preds.clamp(0.0,1.0).cpu(),data_range=1.,downsample=False,mask=masks.cpu(),device='cpu').item())
                    if os.path.exists('./save_image/test/epochs'+str(epoch)) == False:
                        os.makedirs('./save_image/test/epochs'+str(epoch))

            print("##### EVAL #####")
            print('eval loss: {:.6f}'.format(epoch_losses_eval.avg))
            print('bpnn loss: {:.6f}'.format(bpnn_loss_eval.avg))
            print('psnr : {:.6f}'.format(psnr.avg))
            print('ssim : {:.6f}'.format(ssim_list.avg))
            e_score.append(epoch_losses_eval.avg)
            e_bpnn.append(bpnn_loss_eval.avg)
            e_psnr.append(psnr.avg)
            e_ssim.append(ssim_list.avg)
            
            psnr_test = AverageMeter()
            ssim_test = AverageMeter()
            epoch_losses_test = AverageMeter()
            bpnn_loss_test = AverageMeter()
            L_loss_test=np.zeros((len(test_dataloader),args.NB_LABEL))
            IDs=[]
            for i,data in enumerate(test_dataloader):

                inputs, labels, masks, imagename = data
                inputs = inputs.reshape(inputs.size(0),1,inputs.size(2),inputs.size(2))
                labels = labels.reshape(labels.size(0),1,512,512)
                labels_bin = labels_bin.reshape(labels_bin.size(0),1,512,512)
                masks = masks.reshape(masks.size(0),1,512,512)

                inputs = inputs.to(device)
                labels = labels.to(device)
                labels_bin = labels_bin.to(device)
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
                    labels_bin = labels_bin>0.225
                    labels_bin = labels_bin.astype("float32")
                    labels_bin = torch.from_numpy(labels_bin).to(device)
                    P_SR = model_bpnn(masks_bin,preds_bin)
                    P_HR = model_bpnn(masks_bin,labels_bin)
                    BVTV_SR = bvtv_loss(preds_bin,masks)
                    BVTV_HR = bvtv_loss(labels_bin,masks)
                    P_SR = torch.cat((P_SR,BVTV_SR),dim=1)
                    P_HR = torch.cat((P_HR,BVTV_HR),dim=1)
                    Ltest_SR = criterion(preds, labels)
                    Ltest_BPNN = Lbpnn(preds,masks,labels_bin)
                    loss_test = Ltest_SR + (args.alpha[trial] * Ltest_BPNN)

                    if epoch == args.num_epochs - 20 :
                        torchvision.utils.save_image(labels_bin, args.outputs_dir +'/alpha_'+str(args.alpha[trial]) + '/labels_bin_'+imagename[0])
                        torchvision.utils.save_image(labels, args.outputs_dir +'/alpha_'+str(args.alpha[trial])+'/labels_'+imagename[0])
                        torchvision.utils.save_image(preds_bin,args.outputs_dir+'/alpha_'+str(args.alpha[trial])+'/preds_bin_'+imagename[0])
                        torchvision.utils.save_image(preds,args.outputs_dir+'/alpha_'+str(args.alpha[trial])+'/preds'+imagename[0])
                        torchvision.utils.save_image(inputs,args.outputs_dir+'/alpha_'+str(args.alpha[trial])+'/inputs'+imagename[0])
                    epoch_losses_test.update(loss_test.item())
                    bpnn_loss_test.update(Ltest_BPNN.item())
                    psnr_test.update(calc_psnr(labels.cpu(),preds.clamp(0.0,1.0).cpu(),masks.cpu(),device="cpu").item())
                    ssim_test.update(ssim(x=labels.cpu(),y=preds.clamp(0.0,1.0).cpu(),data_range=1.,downsample=False,mask=masks.cpu(),device="cpu").item())
            print("##### TEST #####")
            print("ssim",epoch_losses_test.avg)
            print("psnr", psnr_test.avg)
            
            writer.add_scalars(str(args.alpha[trial])+'/PSNR',{'train':psnr_train.avg,'val':psnr.avg,'test':psnr_test.avg},epoch)
            writer.add_scalars(str(args.alpha[trial])+'/Loss MPNN',{'train':bpnn_loss.avg,'test':bpnn_loss_test.avg,'val':bpnn_loss_eval.avg},epoch)
            writer.add_scalars(str(args.alpha[trial])+'/SSIM',{'train':ssim_train.avg,'test':ssim_test.avg,'val':ssim_list.avg},epoch)
            writer.add_scalars(str(args.alpha[trial])+'/Loss',{'train':epoch_losses.avg,'test':epoch_losses_test.avg,'val':epoch_losses_eval.avg},epoch)

            if epoch_losses_test.avg < best_loss:
                best_epoch = epoch
                best_loss = epoch_losses_test.avg
                best_weights = copy.deepcopy(model.state_dict())

    torch.save(best_weights, os.path.join(args.outputs_dir, 'alpha'+str(args.alpha[trial])+'best.pth'))
    writer.close()
    return np.min(np.array(e_bpnn)),np.max(np.array(e_psnr)),args.alpha[trial],np.max(np.array(e_ssim)),

for n_trial in range(2):
    bp,ps,al,ss = objective(n_trial)

