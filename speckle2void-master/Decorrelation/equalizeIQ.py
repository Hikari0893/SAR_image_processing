import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift


def equalizeIQ(C, mx, my, rx, ry):
    """
    C: single look complex data
    mx, my: frequency shift in normalized frequencies (pi*m)
    rx, ry: only frequencies in [-pi(1-r), pi(1-r)] are equalized
    
    CE: equalized complex data
    W: frequency mask of whitening filter
    """
    r, c = C.shape
    Freq_shift = np.exp(-1j * np.pi * np.arange(r)[:, None] * my) * np.exp(-1j * np.pi * np.arange(c) * mx)
    C2 = Freq_shift * C
    
    fC = fft2(C2)
    PSD = np.real(fC * np.conj(fC))
    
    R = ifft2(PSD)
    rho_x = np.abs(R[0, 1] / R[0, 0])
    rho_y = np.abs(R[1, 0] / R[0, 0])
    
    r1 = round(c * rx)
    x1 = np.arange(c)
    y1 = np.sqrt(np.mean(fftshift(PSD), axis=0))
    clipping = np.percentile(y1, 99)
    y1[y1 > clipping] = clipping
    y1 /= np.max(y1)
    
    if 2 * r1 >= len(x1):
        r1 = len(x1) // 2 - 1

    p = np.polyfit(x1[r1:-r1+1], y1[r1:-r1+1], 70)
    yi1 = np.polyval(p, x1)
    plt.figure()
    plt.plot(x1, y1, '*', x1[r1:-r1+1], yi1[r1:-r1+1], 'o')
    plt.show()
    
    r2 = round(r * ry)
    x2 = np.arange(r)
    y2 = np.sqrt(np.mean(fftshift(PSD), axis=1))
    clipping = np.percentile(y2, 99)
    y2[y2 > clipping] = clipping
    y2 /= np.max(y2)

    if 2 * r2 >= len(x2):
        r2 = len(x2) // 2 - 1
    
    p = np.polyfit(x2[r2:-r2+1], y2[r2:-r2+1], 70)
    yi2 = np.polyval(p, x2)
    plt.figure()
    plt.plot(x2, y2, '*', x2[r2:-r2+1], yi2[r2:-r2+1], 'o')
    plt.show()
    
    G = (yi2[:, None] * yi1[None, :])
    mask = np.zeros((r, c))
    mask[r2:-r2+1, r1:-r1+1] = 1
    G *= mask
    G_norm = np.linalg.norm(G)
    
    if G_norm == 0:
        G_norm = 1  # Prevent division by zero in the next step

    G /= G_norm
    G[G == 0] = np.inf  # Avoid division by zero
    W = 1 / G / np.sqrt(np.sum(mask))
    W[~mask.astype(bool)] = 1
    
    h = np.zeros((r, c))
    h[r2:-r2, r1:-r1] = 1
    h = fftshift(h)
    fC *= h
    
    fCE = fC * W
    CE = ifft2(fCE)
    
    RE = ifft2(PSD * W**2)
    rho_xe = np.abs(RE[0, 1] / RE[0, 0])
    rho_ye = np.abs(RE[1, 0] / RE[0, 0])
    
    return CE, W
