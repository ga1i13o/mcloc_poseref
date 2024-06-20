import einops
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F


def gem(x, p=torch.ones(1)*3, eps: float = 1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
    
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        assert x.shape[2] == x.shape[3] == 1, f"{x.shape[2]} != {x.shape[3]} != 1"
        return x[:, :, 0, 0]
    
    
class FlattenFeatureMaps(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = lambda x: einops.rearrange(x, 'b c h w -> b (c h w)')
    def forward(self, x):
        return self.flatten(x)


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)
