import numpy as np

def sigmoid(x : float) -> float:
    """
    Takes in weighted sum of the inputs and normalizes
    them through between 0 and 1 through a sigmoid function
    """
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x : float) -> float:
    """
    The derivative of the sigmoid function used to
    calculate necessary weight adjustments
    """
    return x * (1 - x)
