import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift, ifft, ifftshift

from hamming_window import hamming_window
from zero_pad_freq import zero_pad_freq

SIZE = []

def data_plot(im, threshold):
    im = np.clip(im, 0, threshold)
    im = im / threshold * 255
    
    return im

# Loading image
picture = np.load('/home/tonix/Documents/Dayana/SAR_CNN_Denoising/slc_065.npy')


# Reconstruct the complex image from its real and imaginary parts.
complex_image = picture[:, :, 0] + 1j * picture[: ,: , 1]
threshold = np.mean(np.abs(complex_image)) + 3 * np.std(np.abs(complex_image))
#Free memory
del picture

SIZE = complex_image.shape
print(SIZE)
# Applying FFT by rocomplexw and FFTshift
fft_img = fftshift(fft(complex_image, axis=1))
#Free memory
del complex_image


# Inverse of Hamming window
M = fft_img.shape[1]
inverse_hamming = 1 / hamming_window(M)
for col in range(fft_img.shape[0]):
    fft_img[col] *= inverse_hamming


# Change the overlap calculation and segmentation for n sublooks
# Snumber_SL = 2  # number of sublooks, change as desired
overlap_factor = 0  # for 30% overlap
c = int((overlap_factor * M)/2)
a = fft_img[:,0:M//2+c]
b = fft_img[:,M//2-c:M]
#Free memory
del fft_img

# 0-padding to preserve image size
a_padded = zero_pad_freq(a,SIZE)
b_padded = zero_pad_freq(b, SIZE)
#Free memory
del a,b

sections = [a_padded, b_padded]
#Free memory
del a_padded, b_padded

# Apply Hamming window to each section
for section in sections:
    M_section = section.shape[1]
    hamming_win = hamming_window(M_section)
    for col in range(section.shape[0]):
        section[col] *= hamming_win


section_A = sections[0]
section_B = sections[1]
#Free memory
del sections


# Applying IFFT by row and IFFTshift for each sublook 
spatial_sectA = ifft(ifftshift(section_A), axis=1)
del section_A

np.save('sublookA.npy', np.stack((np.real(spatial_sectA), np.imag(spatial_sectA)), axis = 2))


#Sublook display
plt.figure()
plt.set_cmap('gray')
plt.title('Sublook A')
plt.xlabel('Range')
plt.ylabel('Azimuth')
plt.imshow(data_plot(np.abs(spatial_sectA), threshold))
plt.show()

#IFFT by row and IFFTshift
spatial_sectB = ifft(ifftshift(section_B), axis=1)

np.save('sublookB.npy', np.stack((np.real(spatial_sectB), np.imag(spatial_sectB)), axis = 2))

#Sublook display
plt.figure()
plt.set_cmap('gray')
plt.title('Sublook B')
plt.xlabel('Range')
plt.ylabel('Azimuth')
plt.imshow(data_plot(np.abs(spatial_sectB), threshold))
plt.show()



#Display of the reconstructed image
plt.figure()
plt.set_cmap('gray')
plt.title('Reconstructed Image')
plt.xlabel('Range')
plt.ylabel('Azimuth')
plt.imshow(data_plot(np.abs((spatial_sectA + spatial_sectB)/2), threshold))
plt.show()




