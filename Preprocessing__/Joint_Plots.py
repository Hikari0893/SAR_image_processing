import numpy as np

def calculate_1D_spectrum_joint(data):
    magnitude_spectrum = np.abs(data)
    del data
    magnitude_spectrum_norm = magnitude_spectrum / np.max(magnitude_spectrum)
    del magnitude_spectrum
    avg_spectrum_row = np.mean(magnitude_spectrum_norm, axis=0)
    del magnitude_spectrum_norm
    return avg_spectrum_row
