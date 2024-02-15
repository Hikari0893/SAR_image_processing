import numpy as np
import matplotlib.pyplot as plt

def hamming_window(N):
    """Generate a Hamming window."""
    n = np.arange(N)
    alpha = 6.00000023841857910e-1
    beta = 1-alpha
    window = alpha - beta * np.cos(2 * np.pi * n / (N-1))
    return window

"""# Example usage:
N = 512
window = hamming_window(N)

# Plotting
plt.plot(window)
plt.title("Hamming Window")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()"""
