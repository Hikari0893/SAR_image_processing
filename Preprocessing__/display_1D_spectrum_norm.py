import numpy as np
import matplotlib.pyplot as plt

def display_1D_spectrum_norm(data, title=''):
    
    #norm_spectrum = data/np.max(np.abs(data))
    
    # Calculate the mean of the magnitude spectrum
    magnitude_spectrum = np.abs(data)
    del data
    
    #Normalize
    magnitude_spectrum_norm = magnitude_spectrum/np.max(magnitude_spectrum)
    del magnitude_spectrum
    #Average across the rows (or columns)
    avg_spectrum_row = np.mean(magnitude_spectrum_norm, axis=0)  # This is for columns. For rows, use axis=1.
    del magnitude_spectrum_norm
    
    plt.figure()
    plt.plot(avg_spectrum_row) # label=f'Mean Value: {magnitude_spectrum:.2f}'
    
    plt.title(f'1D Spectrum: {title}')
    plt.xlabel('Frequency Index')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.savefig(f'{title}.png')
    plt.show()