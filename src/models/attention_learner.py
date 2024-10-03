import torch
import torch.nn as nn

from .base_learner import BaseLearner
from .metric import *
from .processor import *
device=torch.device("cuda")

import torch.nn.init as init

class Attentive(nn.Module):
    def __init__(self, size): 
        super(Attentive, self).__init__()
        self.w = nn.Parameter(torch.rand(size), requires_grad=True)


    def forward(self, x):
        return x @ torch.diag(self.w)


class AttLearner(BaseLearner):
    """Attentive Learner"""
    def __init__(self, metric, processors, nlayers, size, activation):

        super(AttLearner, self).__init__(metric, processors)
        self.nlayers = nlayers
        self.layers = nn.ModuleList()
        for _ in range(self.nlayers):
            self.layers.append(Attentive(size))
        self.activation = activation


    def internal_forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != (self.nlayers - 1):
                x = self.activation(x)
        return x

    def forward(self, features):
        z = self.internal_forward(features)
        # print("z1:",z.shape)
        z = F.normalize(z, dim=1, p=2)
        # print("z2:",z.shape)
        similarities = self.metric(z)
        # print("simi:",similarities.shape)
        for processor in self.processors:
            similarities = processor(similarities)
        similarities = F.relu(similarities)

        # torch.set_printoptions(threshold=np.inf)
        # simi = similarities.to("cuda")
        # with open('simi.txt','a') as file0:
        #     print("similarities!!!:",simi,simi.shape,file=file0)

        
        # #返回归一化后的邻接矩阵，该邻接矩阵有强度值
        # inv_sqrt_degree = 1. / (torch.sqrt(similarities.sum(dim=1, keepdim=False)) + 1e-10)
        # temp = inv_sqrt_degree[:, None] * similarities * inv_sqrt_degree[None, :]

        # torch.set_printoptions(threshold=np.inf)
        # simi = temp.to("cuda")
        # with open('temp.txt','a') as file0:
        #     print("similarities!!!:",simi,simi.shape,file=file0)

        return similarities

 
