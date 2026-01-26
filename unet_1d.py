""" Full assembly of the parts to form the complete network """

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

    
class Unet_1D_tanh(nn.Module):
    def __init__(self,n_channels, n_classes):
        super(Unet_1D_tanh, self).__init__()
        self.n_channels2 = n_channels
        self.n_classes2 = n_classes
        self.fc1 = Conv_1D(self.n_channels2, 160)
        self.fc2 = Conv_1D(160, 240)
        self.fc3 = Conv_1D(240+self.n_channels2, 320)
        self.fc4 = Conv_1D(320, 360)
        self.fc5 = Conv_1D(360+self.n_channels2, 480)
        self.fc6 = Conv_1D(480, 520)
        self.fc7 = Conv_1D(520+self.n_channels2, 600)
        self.outfc = Conv_1D_last_tanh(600, self.n_classes2)
    def forward(self, x_mid):
        x21 = self.fc1(x_mid)
        x22 =  torch.cat([x_mid,self.fc2(x21)], dim=1)
        x23 = self.fc3(x22)
        x24 = torch.cat([x_mid,self.fc4(x23)], dim=1)
        x25 = self.fc5(x24)
        x26 = torch.cat([x_mid,self.fc6(x25)], dim=1)
        x27 = self.fc7(x26)
        logits = self.outfc(x27)
        return logits * 0.1 + 1
    
class Conv_1D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_1d = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            # nn.LeakyReLU(inplace=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv_1d(x)
    


class Conv_1D_last_tanh(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_1d = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        return self.conv_1d(x)


