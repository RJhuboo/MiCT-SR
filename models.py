import torch
import math
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms 

class ThrSigmoid(nn.Module):
    def __init__(self, k,t):
        super().__init__()
        self.k = k
        self.t = t
    def forward(self, x):
        ex = (1/(1+torch.exp(-self.k*(x-self.t))))
        return ex

class FSRCNN(nn.Module):
    def __init__(self, scale_factor, num_channels=1, d=56, s=12, m=4,device='cpu'):
        super(FSRCNN, self).__init__()
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=5, padding=5//2),
            nn.PReLU(d)
        )
        self.mid_part = [nn.Conv2d(d, s, kernel_size=1), nn.GELU()] #PReLU(s)
        for _ in range(m):
            self.mid_part.extend([nn.Conv2d(s, s, kernel_size=3, padding=3//2), nn.GELU()]) #PReLU(s)
        self.mid_part.extend([nn.Conv2d(s, d, kernel_size=1), nn.GELU()]) #PReLU(d)
        self.mid_part = nn.Sequential(*self.mid_part)
        self.last_part = nn.ConvTranspose2d(d, num_channels, kernel_size=9, stride=scale_factor, padding=9//2,
                                            output_padding=scale_factor-1)

        self._initialize_weights()
        self.sig = ThrSigmoid(k=400,t=0.2) 
    def _initialize_weights(self):
        for m in self.first_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        for m in self.mid_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        nn.init.normal_(self.last_part.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.last_part.bias.data)

    def forward(self, x):
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.last_part(x)
        x_bin = self.sig(x.clamp(0.0,1.0))
        return x,x_bin

    
## Neural Network for regression ##
class NeuralNet(nn.Module):
    def __init__(self,n1,n2,n3,out_channels):
        super().__init__()
        self.fc1 = nn.Linear((64*64*64)+(64*64),n1)
        self.fc2 = nn.Linear(n1,n2)
        self.fc3 = nn.Linear(n2,n3)
        #self.fc5 = nn.Linear(n3,20)
        self.fc4 = nn.Linear(n3,out_channels)
    def forward(self,mask,x):
        mask = torch.flatten(mask,1)
        x = torch.flatten(x,1)
        x = torch.cat((x,mask),1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
## 3 CNN model ##
class BPNN(nn.Module):
    def __init__(self,in_channel,features,out_channels,n1=240,n2=120,n3=60,k1=3,k2=3,k3=3):
        super(BPNN,self).__init__()
        # initialize CNN layers 
        self.conv1 = nn.Conv2d(in_channel,features,kernel_size = k1,stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(features,features*2, kernel_size = k2, stride = 1, padding = 1)
        self.conv3 = nn.Conv2d(features*2,64, kernel_size = k3, stride = 1, padding = 1)
        self.pool = nn.MaxPool2d(2,2)
        # initialize NN layers
        #self.fc1 = nn.Linear(64**3,n1)
        #self.fc2 = nn.Linear(n1,n2)
        #self.fc3 = nn.Linear(n2,14)
        self.dropout = nn.Dropout2d(0.25)
        self.neural = NeuralNet(n1,n2,n3,out_channels)
        # dropout
        # self.dropout = nn.Dropout(0.25)
    def forward(self,mask, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x= self.dropout(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.neural(mask,x)
        #x = self.neural(x)
        #x = torch.flatten(x,1)
        return x 
