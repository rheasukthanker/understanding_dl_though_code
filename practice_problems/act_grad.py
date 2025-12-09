import math
def activation_derivatives(x: float) -> dict[str, float]:

    def sigmoid(x):
        return 1/(1+math.exp(-x))
    def tanh(x):
        return (math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x))
    def relu(x):
        return max(0,x)
    act_grad = {}
    act_grad["sigmoid"] = sigmoid(x)*(1-sigmoid(x))
    act_grad["tanh"] = 1-(tanh(x))**2
    if x>0:
        act_grad["relu"]=1
    else:
        act_grad["relu"]=0
    return act_grad