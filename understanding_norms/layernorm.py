import torch


class MyLayerNorm(torch.nn.Module):
    def __init__(self, num_features, eps=1e-05, affine=False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.training = True
        if affine:
            self.gamma = torch.nn.Parameter(torch.ones(num_features))
            self.beta = torch.nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def forward(self, x):
        assert x.shape[-1] == self.num_features
        mean = torch.mean(x, dim=-1)
        var = torch.var(x, dim=-1, unbiased=False)
        if not self.affine:
            return (x - mean[..., None]) / torch.sqrt(var[..., None] + self.eps)
        else:
            return (
                (x - mean[..., None]) / torch.sqrt(var[..., None] + self.eps)
            ) * self.gamma + self.beta


if __name__ == "__main__":
    batch_size = 32
    num_features = 128
    inp = torch.randn([batch_size, 4, num_features])
    mylayer = MyLayerNorm(num_features, affine=True, eps=1e-04)
    layer = torch.nn.LayerNorm(num_features, elementwise_affine=True, eps=1e-04)
    assert torch.allclose(mylayer(inp), layer(inp), atol=1e-05)
    mylayer.eval()
    layer.eval()
    assert torch.allclose(mylayer(inp), layer(inp), atol=1e-05)
