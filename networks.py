import torch
from torch import nn
import torch.nn.functional as F
import torchaudio.functional as TAF
from utils import *
import constants as C

class Model(nn.Module):
    def __init__(self, causal=True):
        super().__init__()
        #ker = 3
        #dilation = 1
        #padding = (dilation * (ker-1) - 1) // 2 + 1
        #self.gru = nn.GRU(1,8, batch_first=True)
        #self.fc = nn.Conv1d(8, 1, 1)
        #self.norm = nn.BatchNorm1d(8)
        #self.conv = CausalConv1d(1, 1, 2, stride=1, dilation=1)
        nch = 32
        self.conv = nn.Sequential(
            CausalConv1d(1, nch, 2, stride=1, dilation=1),
            nn.GELU(),
            CausalConv1d(nch, nch, 2, stride=1, dilation=1),
            nn.GELU(),
            CausalConv1d(nch, nch, 2, stride=1, dilation=1),
            nn.GELU(),
            CausalConv1d(nch, 1, 2, stride=1, dilation=1),
        )

    def forward(self, x):
        #r = x
        #x = x.transpose(2,1)
        #x, _ = self.gru(x)
        #x = torch.relu(x)
        #x = x.transpose(2,1)
        #x = self.norm(x + r)
        #x = self.fc(x)

        y = self.conv(x)

        #return F.relu6(x * y) / 6
        return F.relu6(x + y) / 6

class CausalConv1d(nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)

        self.left_padding = dilation * (kernel_size - 1)

    def forward(self, input):
        x = F.pad(input.unsqueeze(2), (self.left_padding, 0, 0, 0)).squeeze(2)
        return super(CausalConv1d, self).forward(x)

