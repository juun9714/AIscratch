import numpy as np

def mse(y,t):
    return 0.5*np.sum((y-t)**2)