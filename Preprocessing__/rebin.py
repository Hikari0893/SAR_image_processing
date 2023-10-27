import numpy as np



def rebin(a, shape):
    m, n = a.shape
    p, q = shape

    # Ensure the data can be divided evenly into the new shape, otherwise crop
    if m % p != 0 or n % q != 0:
        a = a[:m - m % p, :n - n % q]
    
    # Reshape and compute the mean for each block
    sh = (p, m // p, q, n // q)
    return a.reshape(sh).mean(3).mean(1)
