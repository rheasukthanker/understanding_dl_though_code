from typing import Callable

def compute_hessian(f: Callable[[list[float]], float], point: list[float], h: float = 1e-5) -> list[list[float]]:
    """
    Compute the Hessian matrix using finite differences.
    
    Args:
        f: A scalar function that takes a list of floats
        point: The point at which to compute the Hessian
        h: Step size for finite differences
        
    Returns:
        The Hessian matrix as a list of lists
    """
    n = len(point)
    hessian = [[0.0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            if i == j:
                # Diagonal: f''(x) = (f(x+h) - 2f(x) + f(x-h)) / h^2
                point_plus = point.copy()
                point_minus = point.copy()
                point_plus[i] += h
                point_minus[i] -= h
                hessian[i][j] = (f(point_plus) - 2 * f(point) + f(point_minus)) / (h ** 2)
            else:
                # Off-diagonal: mixed partial using 4-point formula
                point_pp = point.copy()
                point_pm = point.copy()
                point_mp = point.copy()
                point_mm = point.copy()
                
                point_pp[i] += h
                point_pp[j] += h
                point_pm[i] += h
                point_pm[j] -= h
                point_mp[i] -= h
                point_mp[j] += h
                point_mm[i] -= h
                point_mm[j] -= h
                
                hessian[i][j] = (f(point_pp) - f(point_pm) - f(point_mp) + f(point_mm)) / (4 * h ** 2)
    
    return hessian