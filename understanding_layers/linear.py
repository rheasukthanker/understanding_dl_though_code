import torch 
from torch.nn import functional as F, init
import math 
class Linear(torch.nn.Module):
      def __init__(self, input_dim, output_dim, bias=True):
          super().__init__()
          self.input_dim = input_dim 
          self.output_dim = output_dim 
          self.weight = torch.nn.Parameter(torch.empty((output_dim, input_dim)))
          if bias:
             self.bias = torch.nn.Parameter(torch.empty((output_dim,)))
          else:
             self.bias = None 
          init.kaiming_uniform_(self.weight, a=math.sqrt(5))
          if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

      def forward(self,x):
          if self.bias is not None:
             return x@self.weight.T + self.bias
          else:
              return x@self.weight.T

if __name__ == "__main__":
   input_dim = 128
   output_dim = 256
   batch_size = 128
   input = torch.randn([32,input_dim])
   mylinear = Linear(input_dim=input_dim,output_dim=output_dim,bias=True)
   linear = torch.nn.Linear(in_features=input_dim,out_features=output_dim,bias=True)
   out_mylinear = mylinear(input)
   linear.weight.data.copy_(mylinear.weight.data)
   linear.bias.data.copy_(mylinear.bias.data)
   outlinear = linear(input)
   assert torch.allclose(out_mylinear,outlinear,atol=1e-6)
   mylinear = Linear(input_dim=input_dim,output_dim=output_dim,bias=False)
   linear = torch.nn.Linear(in_features=input_dim,out_features=output_dim,bias=False)
   out_mylinear = mylinear(input)
   linear.weight.data.copy_(mylinear.weight.data)
   outlinear = linear(input)
   assert torch.allclose(out_mylinear,outlinear,atol=1e-6)



        



            