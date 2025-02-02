import torch


class MyGroupNorm(torch.nn.Module):
    def __init__(self, num_groups, num_channels, eps=1e-05, affine=False):
        super().__init__()
        assert num_channels % num_groups == 0
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.training = True
        if affine:
            self.gamma = torch.nn.Parameter(torch.ones(num_features)).reshape(
                1, self.num_channels, 1, 1
            )
            self.beta = torch.nn.Parameter(torch.zeros(num_features)).reshape(
                1, self.num_channels, 1, 1
            )
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def forward(self, x):
        assert x.shape[1] == self.num_channels
        B, C, H, W = x.shape
        x = x.reshape(B, self.num_groups, C // self.num_groups, H, W)
        mean = x.mean(dim=(2, 3, 4), keepdim=True)  # Compute mean within each group
        var = x.var(dim=(2, 3, 4), keepdim=True, unbiased=False)  # Compute variance

        x = (x - mean) / (torch.sqrt(var + self.eps))  # Normalize
        x = x.reshape(B, C, H, W)
        if not self.affine:
            return x
        else:
            return x * self.gamma + self.beta


if __name__ == "__main__":
    batch_size = 32
    num_features = 128
    num_groups = 32
    inp = torch.randn([batch_size, num_features, 32, 32])
    mylayer = MyGroupNorm(num_groups, num_features, affine=True, eps=1e-04)
    layer = torch.nn.GroupNorm(num_groups, num_features, affine=True, eps=1e-04)
    assert torch.allclose(mylayer(inp), layer(inp), atol=1e-05)
    mylayer.eval()
    layer.eval()
    assert torch.allclose(mylayer(inp), layer(inp), atol=1e-05)
