import numpy as np
import matplotlib.pyplot as plt

def hamming_window(M):
    # This function returns the Hamming window of size M
    alpha = 6.00000023841857910e-1
    n = np.arange(M)
    w = alpha - (1 - alpha) * np.cos((2 * np.pi * n) / (M - 1))
    
    # Plotting
    plt.plot(w)
    plt.title("Hamming Window")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()
    
    return w