
import torch 
import torch.nn as nn
class CausalAttention(nn.Module):
    def __init__(self, seq_len:int=512, embed_dim:int=64, head_size:int=32):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_size = head_size 
        self.seq_len = seq_len
        self.K = torch.nn.Linear(embed_dim,head_size)
        self.Q = torch.nn.Linear(embed_dim,head_size)
        self.V = torch.nn.Linear(embed_dim,head_size)
        self.mask = torch.ones(seq_len, seq_len).triu(diagonal=1)
        self.mask.masked_fill_(self.mask.bool(), float("-inf"))
        self.mask = self.mask.view(1, 1, *self.mask.shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)
        attention = Q@K.permute(0,2,1)
        attention = attention/torch.sqrt(torch.tensor(self.head_size))
        attention = attention+self.mask
        attention = torch.nn.functional.softmax(attention,dim=-1)/torch.sqrt(torch.tensor(self.head_size))
        return attention@V 
    
class MultiheadAttention(nn.Module):
    r"""
    https://arxiv.org/abs/1706.03762
    """
    def __init__(self, seq_len:int=512, head_size:int = 32, embed_dim: int = 64, n_head:int=8) -> None:
        super().__init__()
        self.n_head = n_head
        self.embed_dim = embed_dim
        self.head_size = head_size
        self.dim_K = torch.tensor(self.head_size)
        self.proj = nn.Parameter(torch.empty(self.head_size*self.n_head, self.embed_dim))
        nn.init.xavier_uniform_(self.proj)
        self.multihead = nn.ModuleList([
           CausalAttention(seq_len, embed_dim, head_size) for _ in range(n_head)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        Z_s = torch.cat([head(x).unsqueeze(2) for head in self.multihead], dim=2)
        Z_s = Z_s.reshape(x.shape[0],x.shape[1],-1)
        Z = torch.matmul(Z_s, self.proj)
        return Z

if __name__=="__main__":
    mha = MultiheadAttention(512,32,64,8)
    input = torch.randn([4,512,64])
    output = mha(input)
    print(output.shape)