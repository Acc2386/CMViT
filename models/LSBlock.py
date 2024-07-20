import torch
import torch.nn as nn
import math
from einops import rearrange


class LSBlock(nn.Module):
    def __init__(self, inp, oup, stride, kernel_size):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]
        expansion = 4
        low = 32
        hidden_dim = int(inp * expansion)

        if stride > 1:
            self.branch1 = nn.Sequential(
                # pw
                nn.Conv2d(inp, low, 1, 1, 0, bias=False),
                nn.BatchNorm2d(low),
                # h_swish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(low, low, kernel_size, stride, (kernel_size - 1) // 2, groups=low,
                          bias=False),
                nn.BatchNorm2d(low),
                # Squeeze-and-Excite
                # SeBlock(low) if use_se else nn.Sequential(),
                # h_swish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(low, oup // 2, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup // 2)
            )
            self.branch2 = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # h_swish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim,
                          bias=False),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                # SeBlock(hidden_dim) if use_se else nn.Sequential(),
                # h_swish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup // 2, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup // 2)
            )

        else:
            self.branch2 = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # h_swish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim,
                          bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # Squeeze-and-Excite
                # SeBlock(hidden_dim) if use_se else nn.Sequential(),
                # h_swish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp),
                nn.SiLU(),
            )

    def forward(self, x):
        if self.stride == 1:
            return self.branch2(x)
        else:
            return torch.cat((self.branch1(x), self.branch2(x)), dim=1)






