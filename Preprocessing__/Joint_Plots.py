import numpy as np

def calculate_1D_spectrum_joint(data, axis=0, norm=False):
    magnitude_spectrum = np.abs(data)
    if norm:
        magnitude_spectrum_norm = magnitude_spectrum / np.max(magnitude_spectrum)
    else:
        magnitude_spectrum_norm = magnitude_spectrum
    avg_spectrum_row = np.mean(magnitude_spectrum_norm, axis=axis)
    return avg_spectrum_row
