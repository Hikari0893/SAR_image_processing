import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft,fftshift, ifft, ifftshift, fft2

def plot_gen(vect,name):
    # Plotting az_t as heatmap
    plt.imshow(np.abs(vect))
    plt.title('Heatmap of az_t')
    plt.xlabel('Range Bins')
    plt.ylabel('Azimuth Bins')
    plt.savefig(name, dpi=300, bbox_inches='tight')

LOWER_LIM = 4500
UPPER_LIMT = 5000

# Load SAR image
picture = np.load('converted_image.npy')

complex_image = picture[LOWER_LIM:UPPER_LIMT, LOWER_LIM:UPPER_LIMT, 0] + 1j * picture[LOWER_LIM: UPPER_LIMT, LOWER_LIM:UPPER_LIMT, 1]
del picture

plot_gen(complex_image,"before_hamming_az.png")

#spectrum in azimuth
spectrum_az0 = fft(complex_image)
del complex_image
spf0 = fftshift(spectrum_az0)
del spectrum_az0



# window size
M = spf0.shape[0] 
# inverse of hamming
dehamming_az = 1/(np.hamming(M))
# freq multiply
delobed_az  = spf0*dehamming_az[:, np.newaxis]
del spf0

#pass to time
az_t = ifft(delobed_az)
del delobed_az
az_t = ifftshift(az_t)

plot_gen(az_t,"after_hamming_az.png")
