import torch
import torch.nn as nn
from math import sqrt
import torch.nn.functional as F




class dense(nn.Module):
    def __init__(self):
        super(dense, self).__init__()
    
        self.res1 = Conv_Res_Block()  
        self.res2 = Conv_Res_Block()  
        self.res3 = Conv_Res_Block()  
        self.res4 = Conv_Res_Block()  
        
    def forward(self, x):
        x1 = x
        x2 = self.res1(x1)# + x1
        x3 = self.res2(x2)# + x2 + x1
        x4 = self.res3(x3)# + x3 + x2 + x1
        x5 = self.res4(x4)# 
        return x5 





class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
        
class Conv_Res_Block(nn.Module):
    def __init__(self):
        super(Conv_Res_Block, self).__init__()
        self.convs = nn.Sequential(
                Conv_ReLU_Block(),
                Conv_ReLU_Block(),
                
                )
    def forward(self, x):
        return self.convs(x) + x
 



class MASK(nn.Module):
    def __init__(self, scale = 2):
        super(MASK, self).__init__()
        self.residual_layer = self.make_layer(Conv_ReLU_Block, 10)
        self.input = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                )
        self.output = nn.Sequential(
                nn.Conv2d(64,64,3,1,1),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
                )
        self.dense_down_1 = dense()
        self.dense_down_2 = dense()
        self.dense_up_1 = dense()
        self.dense_up_2 = dense()
        self.dense_bottom = dense()
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
        # self.sigmoid = nn.Sigmoid() 
    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return (nn.Sequential(*layers))

    def forward(self, x):
        residual = x
        out1 = self.input(x)
        out2 = self.dense_down_1(F.max_pool2d(out1,2))
        out3 = self.dense_down_2(F.avg_pool2d(out2,2))
        out3_ = out3 + self.dense_bottom(out3)
        out2_ = out2 + F.upsample(self.dense_up_2(out3_), scale_factor=2)
        out1_ = out1 + F.upsample(self.dense_up_1(out2_), scale_factor=2)
        out = self.output(out1_)
        # out = torch.add(out,residual)
        return F.sigmoid(out)
