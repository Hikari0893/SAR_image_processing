import numpy as np
import matplotlib.pyplot as plt

def display_histogram(data, title=''):
    
    # Compute the magnitude
    magnitude = np.abs(data)
    del data
    
    # Flatten the 2D magnitude spectrum to a 1D array
    flattened_spectrum = magnitude.flatten()
    del magnitude
    
    # Calculate the average of the spectrum values
    #avg_spectrum = np.mean(flattened_spectrum)
    
    # Plot the histogram
    plt.figure()
    plt.hist(flattened_spectrum, bins=50, color='blue', edgecolor='black')
    plt.title('Histogram of the Magnitude Spectrum')
    plt.xlabel('Magnitude Value')
    plt.ylabel('Count')

    plt.grid(True)
    #plt.savefig(f'Hist{title}.png')
    plt.show()
    