import numpy as np
from scipy import special

# Extract relevant parameters
L = 1
M = 10.089038980848645
m = -1.429329123112601
c = (1 / 2) * (special.psi(L) - np.log(L))

def normalize(batch):
    return (np.log(batch + 1e-7) - 2 * m) / (2 * (M - m))

def denormalize(batch):
    # return np.exp((M - m) * np.clip(np.squeeze(im), 0, 1) + m) + 1e-6
    # return np.exp(((M - m) * np.squeeze(im) + m + c))
    # return np.exp(((M - m) * np.clip(np.squeeze(batch), 0, 1) + m))
    return np.exp(2 * batch * (M - m) + m + c)