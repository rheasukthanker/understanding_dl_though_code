import numpy as np

def apply_temperature(logits, temperature):
    logits = logits / temperature  # Apply temperature scaling
    probs = np.exp(logits) / np.sum(np.exp(logits),axis=-1,keepdims=True)  # Softmax to get probabilities
    return np.array([np.random.choice(np.arange(len(logits[i])), p=probs[i]) for i in range(probs.shape[0])])  # Sample from the distribution

if __name__=="__main__":
    logits = np.random.randn(8,10)*34
    print(apply_temperature(logits,1))
    print(apply_temperature(logits,0.999999999999999999))
    print(apply_temperature(logits,0.999999999999))
    print(apply_temperature(logits,0.9999))
    print(apply_temperature(logits,0.99))
    print(apply_temperature(logits,0.9))
    print(apply_temperature(logits,20))
    print(apply_temperature(logits,100))