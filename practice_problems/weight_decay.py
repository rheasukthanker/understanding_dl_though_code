def apply_weight_decay(parameters: list[list[float]], gradients: list[list[float]], 
                       lr: float, weight_decay: float, apply_to_all: list[bool]) -> list[list[float]]:
    """
    Apply weight decay (L2 regularization) to parameters.
    
    Args:
        parameters: List of parameter arrays (weights and biases)
        gradients: List of gradient arrays corresponding to parameters
        lr: Learning rate
        weight_decay: Weight decay factor (lambda)
        apply_to_all: List of booleans indicating whether to apply weight decay to each parameter group
    
    Returns:
        Updated parameters after applying weight decay
    """
    updated_params = []
    
    for param_array, grad_array, apply_wd in zip(parameters, gradients, apply_to_all):
        updated_array = []
        for param, grad in zip(param_array, grad_array):
            if apply_wd:
                # Apply weight decay: param = param - lr * (grad + weight_decay * param)
                # Which simplifies to: param = param - lr * grad - lr * weight_decay * param
                updated_param = param - lr * grad - lr * weight_decay * param
            else:
                # No weight decay (e.g., for biases)
                updated_param = param - lr * grad
            updated_array.append(updated_param)
        updated_params.append(updated_array)
    
    return updated_params