# -*- coding: utf-8 -*-
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable

#Class to be used for functionality shared by all networks
class networkUtils:
    def __init__(self):
        pass


class GaussianDropout(nn.Module):
    def __init__(self, p=0.5):
        super(GaussianDropout, self).__init__()
        alpha = p/(1-p)
        self.alpha = torch.Tensor([alpha])
        
    def forward(self, x):
#         Sample noise   e ~ N(1, alpha)
        epsilon = Variable(torch.randn(x.size()) * self.alpha + 1)
        if x.is_cuda():
            epsilon = epsilon.cuda()
        return x * epsilon
        
