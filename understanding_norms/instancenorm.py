import torch


class MyInstanceNorm(torch.nn.Module):
    def __init__(self, num_channels, eps=1e-05, affine=False):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.training = True

        if affine:
            self.gamma = torch.nn.Parameter(
                torch.ones(num_channels).reshape(1, num_channels, 1, 1)
            )
            self.beta = torch.nn.Parameter(
                torch.zeros(num_channels).reshape(1, num_channels, 1, 1)
            )
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def forward(self, x):
        assert x.shape[1] == self.num_channels  # Ensure correct number of channels
        B, C, H, W = x.shape

        mean = x.mean(dim=(2, 3), keepdim=True)  # Compute mean per channel per instance
        var = x.var(
            dim=(2, 3), keepdim=True, unbiased=False
        )  # Compute variance per channel per instance

        x = (x - mean) / torch.sqrt(var + self.eps)  # Normalize

        if not self.affine:
            return x
        else:
            return (
                x * self.gamma + self.beta
            )  # Apply learnable parameters if affine=True


if __name__ == "__main__":
    batch_size = 32
    num_features = 128
    inp = torch.randn([batch_size, num_features, 32, 32])

    mylayer = MyInstanceNorm(num_features, affine=True, eps=1e-04)
    layer = torch.nn.InstanceNorm2d(num_features, affine=True, eps=1e-04)

    assert torch.allclose(
        mylayer(inp), layer(inp), atol=1e-05
    )  # Check if output matches
    mylayer.eval()
    layer.eval()
    assert torch.allclose(
        mylayer(inp), layer(inp), atol=1e-05
    )  # Ensure consistency in eval mode
