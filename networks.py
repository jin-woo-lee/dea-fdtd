import torch
from torch import nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, causal=True):
        super().__init__()
        ker = 3
        dilation_1 = 1
        dilation_2 = 2
        padding_1 = (dilation_1 * (ker-1) - 1) // 2 + 1
        padding_2 = (dilation_2 * (ker-1) - 1) // 2 + 1
        self.gru = nn.GRU(1,8, batch_first=True)
        self.fc = nn.Sequential(
            nn.Conv1d(8, 16, 1),
            nn.ReLU(),
            nn.Conv1d(16, 16, 1),
            nn.ReLU(),
            nn.Conv1d(16, 1, 1),
        )
        self.conv = CausalConv1d(1, 1, 2, stride=1, dilation=1)

    def forward(self, x):
        x = x.transpose(2,1)
        x, _ = self.gru(x)
        x = torch.relu(x)
        x = x.transpose(2,1)
        x = self.fc(x)

        x = self.conv(x)
        return x

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

