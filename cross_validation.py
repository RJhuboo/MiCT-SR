from torch.autograd import Variable
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
from skimage.filters import threshold_otsu 
import pytorch_ssim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from models import FSRCNN, BPNN
from datasets import TrainDataset, TestDataset
from utils import AverageMeter, calc_psnr
from ssim import ssim
import time
from PIL import Image


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
    parser.add_argument('--HR_dir', type=str,default = "/gpfsstore/rech/tvs/uki75tv/data_fsrcnn/HR/Train_Label_trab_100")
    parser.add_argument('--LR_dir', type=str,default = "/gpfsstore/rech/tvs/uki75tv/data_fsrcnn/LR/Train_trab")
    parser.add_argument('--mask_dir',type=str,default = "/gpfsstore/rech/tvs/uki75tv/data_fsrcnn/HR/Train_trab_mask")
    parser.add_argument('--tensorboard_name',type=str,default = "rescale")
    parser.add_argument('--outputs-dir', type=str, default = "./FSRCNN_search")
    parser.add_argument('--checkpoint_bpnn', type= str, default = "./checkpoints_bpnn/BPNN_checkpoint_TFfsrcnn.pth")
    parser.add_argument('--alpha', type = list, default = [1e-5])
    parser.add_argument('--Loss_bpnn', default = MSELoss)
    parser.add_argument('--weights-file', type=str)
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)#-2
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=70)
    parser.add_argument('--num-workers', type=int, default=24)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--nof', type= int, default = 64)
    parser.add_argument('--n1', type=int,default = 158)
    parser.add_argument('--n2', type=int,default = 152)
    parser.add_argument('--n3', type=int,default = 83)
    parser.add_argument('--gpu_ids', type=list, default = [0, 1, 2])
    parser.add_argument('--NB_LABEL', type=int, default = 7)
    parser.add_argument('--k_fold', type=int, default = 1)
    parser.add_argument('--name', type=str, default = "BPNN_7lrhr_1alpha_clamp")
    args = parser.parse_args()
    
    ## Create summary for tensorboard
    writer = SummaryWriter(log_dir='runs/'+args.tensorboard_name)
    
    args.outputs_dir = os.path.join(args.outputs_dir, args.name)    
    if os.path.exists(args.outputs_dir) == False:
        os.makedirs(args.outputs_dir)
    if os.path.exists(args.outputs_dir + "/alpha_"+str(args.alpha[trial])) == False:
        os.makedirs(args.outputs_dir + "/alpha_"+str(args.alpha[trial]))
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
        kf = train_test_split(index,train_size=60,test_size=11,shuffle=True,random_state=42)
        kf[0] = sorted(kf[0])
        new_kf = [list(range(kf[0][i]*100,(kf[0][i]+1)*100)) for i in range(60)]
        new_kf=np.array(new_kf)
        kf[0] = np.vstack(new_kf).reshape((6000,1)).flatten()
        new_kf = [list(range(kf[1][i]*100,(kf[1][i]+1)*100)) for i in range(11)]
        new_kf = np.array(new_kf)
        kf[1] = np.vstack(new_kf).reshape((1100,1)).flatten()
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
        #my_transforms = transforms.Compose([
        #  transforms.ToPILImage(),
        #  transforms.RandomRotation(degrees=45),
        #  transforms.RandomHorizontalFlip(p=0.3),
        #  transforms.RandomVerticalFlip(p=0.3),
        #  transforms.RandomAffine(degrees=(0,1),translate=(0.1,0.1)),
        #  transforms.ToTensor(),
        #])
        
        dataset = TrainDataset(args.HR_dir, args.LR_dir, args.mask_dir,transform = my_transforms)
        dataset_test = TestDataset("/gpfsstore/rech/tvs/uki75tv/Test_trab","/gpfsstore/rech/tvs/uki75tv/data_fsrcnn/LR/Test_trab","/gpfsstore/rech/tvs/uki75tv/Test_trab_mask")
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
        # best_weights = copy.deepcopy(model.state_dict())
        # best_epoch = 0
        # best_loss = 10
        # tr_psnr, tr_score, tr_bpnn, tr_ssim = [],[], [],[]
        e_score, e_bpnn, e_psnr,e_ssim = [], [], [], []
        # t_score, t_bpnn,t_psnr,t_ssim = [],[],[],[]
        #start = time.time()
        # best_epoch_tracking = 1000
        # data_param_SR = []
        # data_param_HR = []
        # data_param_SR_test = []
        # data_param_HR_test = []
        # names_index = []
        # names_index_test = []
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
                    #inputs, labels, masks = inputs.to(device), labels.to(device), masks.to(device)
                    inputs, labels, masks = inputs.to(device), labels.to(device), masks.to(device)
                    #print("max labels",torch.max(labels))
                    #print(" min labels", torch.min(labels))
                   #torchvision.utils.save_image(inputs,'inputs.png')
                    #preds = model(inputs)
                    preds,preds_bin = model(inputs)
                    preds = preds.clamp(0.0,1.0)
                    #torchvision.utils.save_image(preds,'preds.png')
                    gaussian_blur = transforms.GaussianBlur((3,3),3)
                    labels_bin = labels.clone().detach()
                    masks_bin = masks.clone().detach()
                    masks_bin = F.interpolate(masks_bin, size=64)
                    labels_bin = gaussian_blur(labels_bin)
                    #preds_bin = preds.clone().cpu().detach().numpy()
                    labels_bin = labels_bin.cpu().numpy()
                    #t1, t2 = threshold_otsu(preds_bin*255),threshold_otsu(labels_bin*255)
                    #print(torch.max(preds),torch.min(preds))
                    #print(torch.max(labels),torch.min(labels))
                    #preds_bin, labels_bin = preds_bin>0.2, labels_bin>0.2
                    labels_bin = labels_bin>0.2
                    #print(np.shape(preds))
                    #im=Image.fromarray(preds_bin.reshape((512,512)) * 255)
                    #im.save("image.png")
                    labels_bin = labels_bin.astype("float32")
                    #preds_bin = preds_bin.astype("float32")
                    #preds_bin = torch.from_numpy(preds_bin).to(device)
                    labels_bin = torch.from_numpy(labels_bin).to(device)
                    #torchvision.utils.save_image(labels_bin, './save_image/labels_bin_'+imagename[0]+'.png')
                    #torchvision.utils.save_image(labels,'./save_image/labels_'+imagename[0]+'.png')
                    #torchvision.utils.save_image(preds_bin,'./save_image/preds_bin_'+imagename[0]+'.pngn')
                    #preds_bin = (preds>0.5).type(torch.float32)
                    P_SR = model_bpnn(masks_bin,preds_bin)
                    P_HR = model_bpnn(masks_bin,labels_bin)
                    BVTV_SR = bvtv_loss(preds_bin,masks)
                    #print("bvtv on SR:",BVTV_SR)
                    BVTV_HR = bvtv_loss(labels,masks)
                    
                    #print("bvtv on HR:",BVTV_HR)
                    P_SR = torch.cat((P_SR,BVTV_SR),dim=1)
                    P_HR = torch.cat((P_HR,BVTV_HR),dim=1)
                    #if epoch == args.num_epochs - 1:
                    #    data_param_SR.append(P_SR.detach().numpy())
                    #    data_param_HR.append(P_HR.detach().numpy())
                    #    names_index.append(imagename)
                    L_SR = criterion(preds, labels)
                    L_BPNN = Lbpnn(P_SR,P_HR)
                    #print("\n max bin:",preds_bin)
                    #print("min bin",torch.min(preds_bin))
                    #print("prediction:", preds_bin)
                    #if epoch >3:
                       #args.alpha[trial]= 0.0001
                    loss = L_SR + (args.alpha[trial] * L_BPNN)
                    epoch_losses.update(loss.item())
                    bpnn_loss.update(L_BPNN.item())
                    optimizer.zero_grad()
                    loss.mean().backward()
                    #if math.isnan(loss.item()) == True: 
                    #for name, param in model.named_parameters():
                    #    if param.requires_grad:
                    #        print(name, param.grad) 
                    #L_SR.mean().backward()
                    optimizer.step()

                    with torch.no_grad():
                        psnr_train.update(calc_psnr(labels.cpu(),preds.clamp(0.0,1.0).cpu(),masks.cpu(),device="cpu").item())
                        ssim_train.update(ssim(x=labels.cpu(),y=preds.clamp(0.0,1.0).cpu(),data_range=1.,downsample=False,mask=masks.cpu(),device="cpu").item())
                    
                    t.set_postfix(loss='{:.9f}'.format(epoch_losses.avg),LossSR='{:.9f}'.format(L_SR.item()),bpnn='{:.3f}'.format(bpnn_loss.avg),psnr='{:.1f}'.format(psnr_train.avg),ssim='{:.1f}'.format(ssim_train.avg),alpha='{:.8f}'.format(args.alpha[trial]))
                    t.update(len(inputs))
                        
            #if epoch > args.num_epochs - 10:
                #torch.save(model.state_dict(), os.path.join(args.outputs_dir, str(args.alpha[trial])+'_best.pth'))
            # tr_psnr.append(psnr_train.avg)
            # tr_ssim.append(ssim_train.avg)
            # tr_score.append(epoch_losses.avg)
            # tr_bpnn.append(bpnn_loss.avg)
            #if epoch==args.num_epochs-1:
            #    df_SR = pd.DataFrame(data_param_SR.reshape(6000,8),index=np.array(names_index).reshape(6000,1))
            #    df_HR = pd.DataFrame(data_param_HR.reshape(6000,8),index=np.array(names_index).reshape(6000,1))
            print("##### Train #####")
            print("Alpha =", args.alpha[trial])
            print("BPNN loss: {:.6f}".format(bpnn_loss.avg))
            print("train loss : {:.6f}".format(epoch_losses.avg))
            #torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))
            #del psnr_train,ssim_train,epoch_losses,bpnn_loss
            
            psnr = AverageMeter()
            ssim_list = AverageMeter()
            model.eval()
            epoch_losses_eval = AverageMeter()
            bpnn_loss_eval = AverageMeter()
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
                    preds=model(inputs)
                    preds,preds_bin = model(inputs)
                    preds=preds.clamp(0.0,1.0)
                    gaussian_blur = transforms.GaussianBlur((3,3),3)
                    labels_bin = labels.clone().detach()
                    labels_bin = gaussian_blur(labels_bin)
                    #preds_bin = preds.clone().cpu().detach().numpy()
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
                    P_SR = torch.cat((P_SR,BVTV_SR),dim=1)
                    P_HR = torch.cat((P_HR,BVTV_HR),dim=1)
                    
                    Leval_SR = criterion(preds, labels)
                    Leval_BPNN = Lbpnn(P_SR,P_HR)
                    
                    loss_eval = Leval_SR + (args.alpha[trial] * Leval_BPNN)
                    epoch_losses_eval.update(loss_eval.item())
                    bpnn_loss_eval.update(Leval_BPNN.item())
                    psnr.update(calc_psnr(labels.cpu(),preds.clamp(0,1).cpu(),masks.cpu(),device="cpu").item())
                    ssim_list.update(ssim(x=labels.cpu(),y=preds.clamp(0.0,1.0).cpu(),data_range=1.,downsample=False,mask=masks.cpu(),device='cpu').item())
                    #if os.path.exists('./save_image/epochs'+str(epoch)) == False:
                    #    os.makedirs('./save_image/epochs'+str(epoch))
                    #torchvision.utils.save_image(labels, './save_image/epochs' + str(epoch) + '/label'+imagename[0]+'.png')
                    #torchvision.utils.save_image(inputs,'./save_image/epochs'+str(epoch) +'/input_'+imagename[0]+'.png')
                    #torchvision.utils.save_image(preds,'./save_image/epochs' + str(epoch) +'/preds'+imagename[0]+'.png')


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
            for i,data in enumerate(test_dataloader):
                inputs, labels, masks, imagename = data
                inputs = inputs.reshape(inputs.size(0),1,256,256)
                labels = labels.reshape(labels.size(0),1,512,512)
                masks = masks.reshape(masks.size(0),1,512,512)
                #inputs, labels, masks = inputs.float(), labels.float(), masks.float()

                inputs = inputs.to(device)
                labels = labels.to(device)
                masks = masks.to(device)
                with torch.no_grad():
                    #preds=model(inputs)
                    preds,preds_bin = model(inputs)
                    preds = preds.clamp(0.0,1.0)
                    gaussian_blur = transforms.GaussianBlur((3,3),3)
                    labels_bin = labels.clone().detach()
                    labels_bin = gaussian_blur(labels_bin)
                    #preds_bin = preds.clone().cpu().detach().numpy()
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
                    P_SR = torch.cat((P_SR,BVTV_SR),dim=1)
                    P_HR = torch.cat((P_HR,BVTV_HR),dim=1)
                    #if epoch==args.num_epochs-1:
                    #    data_param_HR_test.append(P_HR.detach().numpy())
                    #    data_param_SR_test.append(P_SR.detach().numpy())
                    #    names_index_test.append(imagename)
                    Ltest_SR = criterion(preds, labels)
                    Ltest_BPNN = Lbpnn(P_SR,P_HR)
                    loss_test = Ltest_SR + (args.alpha[trial] * Ltest_BPNN)
                    #if epoch == args.num_epochs - 1 :
                    #    torchvision.utils.save_image(labels_bin, args.outputs_dir +'/alpha_'+str(args.alpha[trial]) + '/labels_bin_'+imagename[0])
                    #    torchvision.utils.save_image(labels, args.outputs_dir +'/alpha_'+str(args.alpha[trial])+'/labels_'+imagename[0])
                    #    torchvision.utils.save_image(preds_bin,args.outputs_dir+'/alpha_'+str(args.alpha[trial])+'/preds_bin_'+imagename[0])
                    #    torchvision.utils.save_image(preds,args.outputs_dir+'/alpha_'+str(args.alpha[trial])+'/preds'+imagename[0])
                    
                    epoch_losses_test.update(loss_test.item())
                    bpnn_loss_test.update(Ltest_BPNN.item())
                    psnr_test.update(calc_psnr(labels.cpu(),preds.clamp(0.0,1.0).cpu(),masks.cpu(),device="cpu").item())
                    ssim_test.update(ssim(x=labels.cpu(),y=preds.clamp(0.0,1.0).cpu(),data_range=1.,downsample=False,mask=masks.cpu(),device="cpu").item())
            print("##### TEST #####")
            print("ssim",epoch_losses_test.avg)
            print("psnr", psnr_test.avg)
            # t_score.append(epoch_losses_test.avg)
            # t_bpnn.append(bpnn_loss_test.avg)
            # t_psnr.append(psnr_test.avg)
            # t_ssim.append(ssim_test.avg)
            
            writer.add_scalars(str(args.alpha[trial])+'/PSNR',{'train':psnr_train.avg,'val':psnr.avg,'test':psnr_test.avg},epoch)
            writer.add_scalars(str(args.alpha[trial])+'/Loss MPNN',{'train':bpnn_loss.avg,'test':bpnn_loss_test.avg,'val':bpnn_loss_eval.avg},epoch)
            writer.add_scalars(str(args.alpha[trial])+'/SSIM',{'train':ssim_train.avg,'test':ssim_test.avg,'val':ssim_list.avg},epoch)
            writer.add_scalars(str(args.alpha[trial])+'/Loss',{'train':epoch_losses.avg,'test':epoch_losses_test.avg,'val':epoch_losses_eval.avg},epoch)

            #df_param_SR_test = pd.DataFrame(np.array(data_param_SR_test).reshape(1100,8),index=np.array(names_index_test).reshape(1100,1))
            #df_param_HR_test = pd.DataFrame(np.array(data_param_HR_test).reshape(1100,8),index=np.array(names_index_test).reshape(1100,1))

          #  if epoch_losses_test.avg < best_loss:
          #      best_epoch = epoch
          #      best_loss = epoch_losses_test.avg
                #best_weights = copy.deepcopy(model.state_dict())
         #   del epoch_losses_test, bpnn_loss_test, psnr, ssim_list
        #end = time.time() 
        #print("Time :", end-start) 
        #cross_bpnn = cross_bpnn + np.array(t_bpnn)
        #cross_score = cross_score + np.array(t_score)
        #cross_psnr = cross_psnr + np.array(t_psnr)
        #cross_ssim = cross_ssim + np.array(t_ssim)
        #cross_bpnn_train = cross_bpnn_train +np.array(tr_bpnn)
        #cross_score_train = cross_score_train + np.array(tr_score)
        #cross_psnr_train = cross_psnr_train +np.array(tr_psnr)
        #cross_ssim_train = cross_ssim_train + np.array(ssim_psnr)
        #del tr_psnr, tr_ssim, tr_score,tr_bpnn, t_bpnn,t_score,t_psnr,t_ssim
                  
    #training_info = {"loss_train": cross_psnr_train/args.k_fold,
     #                "loss_val": cross_score/args.k_fold,
     #                "bpnn_train" :cross_bpnn_train/args.k_fold,
     #                "bpnn_val": cross_bpnn/args.k_fold,
     #                "psnr": cross_psnr/args.k_fold,
     #                "ssim":cross_ssim/args.k_fold,
     #                "train_ssim": cross_ssim_train/args.k_fold,
     #                "train_psnr": cross_psnr_train/args.k_fold
     #               }
    # training_info = {"loss_train": tr_score,
    #                  "loss_test": t_score,
    #                  "bpnn_train" : tr_bpnn,
    #                  "bpnn_test": t_bpnn,
    #                  "psnr_test": t_psnr,
    #                  "ssim_test": t_ssim,
    #                  "train_ssim": tr_ssim,
    #                  "train_psnr": tr_psnr,
    #                  "eval_psnr": e_psnr,
    #                  "eval_ssim": e_ssim,
    #                  "bpnn_val": e_bpnn,
    #                  "loss_val": e_score,
    #                  "alpha": args.alpha[trial]
    #                 }
    # print(" ------------ CROSS RESULTS -------------")
    # print("loss train:",training_info["loss_train"])
    # print("loss val:",training_info['loss_val'])
    # print("loss test:",training_info['loss_test'])
    # print("bpnn train:",training_info["bpnn_train"])
    # print("bpnn_val:", training_info["bpnn_val"])
    # print("alpha:",training_info["alpha"])
    #i=1
    #while os.path.exists(os.path.join(args.outputs_dir,"losses_info"+str(i)+".pkl")) == True:
    #    i=i+1
    #with open( os.path.join(args.outputs_dir,"losses_info"+str(i)+".pkl"), "wb") as f:
    #    pickle.dump(training_info,f)
    #print('best epoch: {}, loss: {:.6f}'.format(best_epoch, best_loss))
    #return np.min(np.array(cross_bpnn)/args.k_fold), np.max(np.array(cross_psnr)/args.k_fold), args.alpha[trial], np.max(np.array(cross_ssim)/args.k_fold)
    writer.add_scalar('MPNN',np.min(np.array(e_bpnn)),args.alpha[trial])
    writer.add_scalar('SSIM',np.max(np.array(e_ssim)),args.alpha[trial])
    writer.add_scalar('PSNR',np.max(np.array(e_psnr)),args.alpha[trial])
    return np.min(np.array(e_bpnn)),np.max(np.array(e_psnr)),args.alpha[trial],np.max(np.array(e_ssim)),
    #torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))

# study= {"bpnn" :[], "psnr": [], "alpha": [],"ssim":[]}
for n_trial in range(1):
    bp,ps,al,ss = objective(n_trial)
    # study["bpnn"].append(bp)
    # study["psnr"].append(ps)
    # study["alpha"].append(al)
    # study["ssim"].append(ss)
    
    #with open("BPNN_1.pkl","wb") as f:
    #    pickle.dump(study,f)
