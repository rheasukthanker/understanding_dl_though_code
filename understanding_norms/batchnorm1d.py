
# Batch Norm 1d
import torch

class MyBatchNorm1d(torch.nn.Module):
  def __init__(self, num_features, eps=1e-09, momentum=0.1, affine=True, track_running_stats=True):
      super().__init__()
      self.num_features = num_features
      self.training = True
      self.eps = eps
      self.momentum = momentum
      self.affine = affine
      self.track_running_stats = track_running_stats
      self.register_buffer("running_mean", torch.zeros(num_features))
      self.register_buffer("running_var", torch.ones(num_features))
      if self.affine:
        self.beta = torch.nn.Parameter(torch.zeros(num_features))
        self.gamma = torch.nn.Parameter(torch.ones(num_features))

  def forward(self, x):
      assert x.shape[-1] == self.num_features
      if self.training:
        sample_mean = torch.mean(x, dim=0)
        sample_var = torch.var(x, dim=0, unbiased=False) # do not apply bessel's correction
        if self.track_running_stats:
           self.update_running_mean_and_var(sample_mean, sample_var, x.shape[0]) # use these stats to update running stats
        if self.affine:
          return ((x - sample_mean) / torch.sqrt(sample_var + self.eps)) * self.gamma + self.beta
        else:
          return (x - sample_mean) / torch.sqrt(sample_var + self.eps)
      else:
        return self.forward_test(x)

  def forward_test(self, x):

    if self.affine:
      return ((x - self.running_mean) / torch.sqrt(self.running_var + self.eps)) * self.gamma + self.beta
    else:
      return (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)

  def update_running_mean_and_var(self, mean, var, num_samples):
     if num_samples != 1:
      correction_factor = num_samples / (num_samples - 1) # bessel's correction
     else:
      correction_factor = 1
     # compute exponential moving average of mean and var
     self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean 
     self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var*correction_factor # apply correction for variance


  def reset_running_mean_and_var(self):
    # reset running statistics 
    self.running_mean = torch.zeros(self.num_features)
    self.running_var = torch.ones(self.num_features)

  def eval(self):
    self.training = False

  def train(self):
    self.training = True

if __name__=="main":
    num_features = 32
    batch_size = 128
    mybatchnorm = MyBatchNorm1d(num_features=num_features, momentum=0.1,eps=1e-6)
    train_input = torch.exp(torch.randn([batch_size, num_features]))+torch.sin(torch.randn([batch_size,num_features]))
    batchnorm = torch.nn.BatchNorm1d(num_features, momentum=0.1, affine=True, eps=1e-6)
    # train time
    mybatchnorm.train()
    batchnorm.train()
    assert torch.allclose(mybatchnorm(train_input), batchnorm(train_input), atol=1e-05)
    # test time
    test_input = torch.exp(torch.randn([batch_size, num_features]))+torch.sin(torch.randn([batch_size,num_features]))
    mybatchnorm.eval()
    batchnorm.eval()
    assert torch.allclose(mybatchnorm(test_input), batchnorm(test_input), atol=1e-05)