import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

class MyDropout(nn.Module):
    def __init__(self, p=0.2):
        super(MyDropout, self).__init__()
        self.p = p
        # multiplier is 1/(1-p). Set multiplier to 0 when p=1 to avoid error...
        if self.p < 1:
            self.multiplier_ = 1.0 / (1.0-p)
        else:
            self.multiplier_ = 0.0
    def forward(self, input):
        # if model.eval(), don't apply dropout
        if not self.training:
            return input
        
        # So that we have `input.shape` numbers of Bernoulli(1-p) samples
        selected_ = torch.Tensor(input.shape).uniform_(0,1)>self.p
            
        # Multiply output by multiplier as described in the paper [1]
        return torch.mul(selected_,input) * self.multiplier_
    

if __name__ == "__main__":
    input_dim = 10
    output_dim = 20
    input = torch.randn([32,20])*100
    dropout = MyDropout()
    dropout.training = True
    print(torch.sum(dropout(input)==0,dim=-1))
    output_torch = F.dropout(input,p=0.2)
    print(torch.sum(output_torch==0,dim=-1))
     
        

