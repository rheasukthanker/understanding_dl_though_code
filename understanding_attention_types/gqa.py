
import torch 
import torch.nn as nn
class CausalAttention(nn.Module):
    def __init__(self, seq_len:int=512, embed_dim:int=64, head_size:int=32, n_head:int=8, n_query_groups:int=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_size = head_size 
        self.n_head = n_head
        self.n_query_groups = n_query_groups
        self.seq_len = seq_len
        self.K = torch.nn.Linear(embed_dim,head_size*n_query_groups)
        self.Q = torch.nn.Linear(embed_dim,head_size*n_head)
        self.V = torch.nn.Linear(embed_dim,head_size*n_query_groups)
        self.mask = torch.ones(seq_len, seq_len).triu(diagonal=1)
        self.mask.masked_fill_(self.mask.bool(), float("-inf"))
        self.mask = self.mask.view(1, 1, *self.mask.shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B,T,_ = x.shape
        self.num_queries_per_group = self.n_head//self.n_query_groups
        Q = self.Q(x).reshape(B,T,self.n_head,self.head_size).reshape(B,T,-1)
        K = self.K(x).reshape(B,T,self.n_query_groups,self.head_size).repeat(1,1,self.num_queries_per_group,1).reshape(B,T,-1)
        V = self.V(x).reshape(B,T,self.n_query_groups,self.head_size).repeat(1,1,self.num_queries_per_group,1).reshape(B,T,-1)
        attention = Q@K.permute(0,2,1)
        attention = attention+self.mask
        attention = torch.nn.functional.softmax(attention,dim=-1)/torch.sqrt(torch.tensor(self.head_size))
        return attention@V 
    
class GroupedQueryAttention(nn.Module):
    r"""
    https://arxiv.org/abs/1706.03762
    """
    def __init__(self, seq_len:int=512, head_size:int = 32, embed_dim: int = 64, n_head:int=8, n_query_groups:int=4) -> None:
        super().__init__()
        self.n_head = n_head
        self.embed_dim = embed_dim
        self.head_size = head_size
        self.n_query_groups = n_query_groups
        self.dim_K = torch.tensor(self.head_size)
        self.proj = nn.Parameter(torch.empty(self.head_size*self.n_head, self.embed_dim))
        nn.init.xavier_uniform_(self.proj)
        self.gqa = CausalAttention(seq_len, embed_dim, head_size, n_head, n_query_groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        Z_s = self.gqa(x)
        Z_s = Z_s.reshape(x.shape[0],x.shape[1],-1)
        Z = torch.matmul(Z_s, self.proj)
        return Z

if __name__=="__main__":
    mha = GroupedQueryAttention(512,32,64,8)
    input = torch.randn([4,512,64])
    output = mha(input)
    print(output.shape)