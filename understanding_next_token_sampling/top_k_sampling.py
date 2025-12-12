import torch 
import numpy as np

# numpy
def top_k_sampling_numpy(logits,k):
    top_k_indices = np.argsort(logits,axis=-1)[:,-k:]
    top_k_logits = np.take_along_axis(logits,top_k_indices,axis=-1)
    top_k_probs = np.exp(top_k_logits)/np.sum(np.exp(top_k_logits),axis=-1,keepdims=True)
    selected_index = np.array([
    np.random.choice(top_k_indices[i].reshape(-1), p=top_k_probs[i].reshape(-1))
    for i in range(top_k_probs.shape[0])
])
    return selected_index

# torch
def top_k_sampling_torch(logits,k):
    top_k_indices = torch.argsort(logits,dim=-1)[:,-k:]
    top_k_logits = logits.gather(dim=-1, index=top_k_indices)
    top_k_probs = torch.exp(top_k_logits)/torch.sum(torch.exp(top_k_logits),dim=-1).unsqueeze(-1)
    print(top_k_probs.shape)
    selected_index = torch.multinomial(torch.squeeze(top_k_probs),num_samples=1)
    return selected_index.T

if __name__=="__main__":

    logits_np = np.random.randn(2,10)
    k = 5
    sampled_indexes = top_k_sampling_numpy(logits_np,k)
    logits_torch = torch.tensor(logits_np)
    sampled_indexes_torch = top_k_sampling_torch(logits_torch,k)
    print(logits_torch)
    print(sampled_indexes_torch)
    print(sampled_indexes)


