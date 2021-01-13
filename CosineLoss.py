import torch
import numpy as np


class Cosine_Loss(torch.nn.Module):
    def __init__(self,class_num=10,label_shift=0.3):
        super(Cosine_Loss, self).__init__()
        self.class_num = class_num
        self.label_shift = label_shift
        self.boundary = (np.sqrt(self.class_num+(self.class_num-1)*self.class_num*(self.label_shift**2)))/np.sqrt(self.class_num)
        self.boundary = torch.tensor(self.boundary).cuda()
    def forward(self,predicts,labels):
        class_num = self.class_num
        Batch_size = labels.size(0)
        labels = labels.view(Batch_size,1)
        one_hot_labels = torch.zeros(Batch_size,class_num).scatter_(1,labels.cpu(),1)
        one_hot_labels = one_hot_labels.view(Batch_size,class_num,1).cuda()
        
        one_hot_labels = one_hot_labels*(1-self.label_shift)+self.label_shift

        predicts = torch.nn.functional.normalize(predicts)
        predicts = predicts.view(Batch_size,1,class_num)
        loss = torch.mean(-torch.log(torch.matmul(predicts,one_hot_labels)/self.boundary/2+0.5))
        return loss