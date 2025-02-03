# RMSNorm
import torch


class MyRMSNorm(torch.nn.Module):
    def __init__(self, num_features, eps=1e-05, elementwise_affine=False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.training = True
        if self.elementwise_affine:
            self.gamma = torch.nn.Parameter(torch.ones(num_features))

    def forward(self, x):
        assert x.shape[-1] == self.num_features
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        if self.elementwise_affine:
            return x / rms * self.gamma
        else:
            return x/rms

    def eval(self):
        self.training = False

    def train(self):
        self.training = True


if __name__ == "__main__":
    batch_size = 32
    num_features = 128
    myrms = MyRMSNorm(num_features, elementwise_affine=True, eps=1e-09)
    inp = torch.randn([batch_size, num_features])
    rms = torch.nn.RMSNorm(num_features, elementwise_affine=True, eps=1e-09)
    assert torch.allclose(myrms(inp), rms(inp), atol=1e-05)
    myrms.eval()
    rms.eval()
    assert torch.allclose(myrms(inp), rms(inp), atol=1e-05)
