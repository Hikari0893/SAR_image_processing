import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift, ifft, ifftshift
from scipy.ndimage import gaussian_filter

from hamming_window import hamming_window
from zero_pad_freq import zero_pad_freq
from Preprocessing__.Joint_Plots import calculate_1D_spectrum_joint

SIZE = []
perform_shift = 0

def data_plot(im, threshold):
    im = np.clip(im, 0, threshold)
    im = im / threshold * 255
    
    return im

# Loading image
# picture = np.load('/home/tonix/Documents/Dayana/SAR_CNN_Denoising/slc_065.npy')
picture = np.load('../data/converted_image.npy')

rg_sta = 1000
rg_end = 2000
az_sta = 1000
az_end = 2000
crop = [rg_sta, rg_end, az_sta, az_end]

# Reconstruct the complex image from its real and imaginary parts.
debug = False
full_image = False
if full_image:
    complex_image = picture[:, :, 0] + 1j * picture[: ,: , 1]
else:
    complex_image = picture[rg_sta:rg_end, az_sta:az_end, 0] + 1j * picture[rg_sta:rg_end, az_sta:az_end, 1]

threshold = np.mean(np.abs(complex_image)) + 3 * np.std(np.abs(complex_image))
#Free memory
del picture

SIZE = complex_image.shape
print(SIZE)
# Applying FFT by rocomplexw and FFTshift
fft_img = fftshift(fft(complex_image, axis=1))

# Inverse of Hamming window
M = fft_img.shape[1]
inverse_hamming = 1 / hamming_window(M)
fft_img = fft_img * inverse_hamming[np.newaxis, :]

# Change the overlap calculation and segmentation for n sublooks
# Snumber_SL = 2  # number of sublooks, change as desired
overlap_factor = 0  # for 30% overlap
c = int((overlap_factor * M)/2)
a = fft_img[:,0:M//2+c]
b = fft_img[:,M//2-c:M]

# if perform_shift:
#     a_shift = np.argmax(np.gradient(gaussian_filter(calculate_1D_spectrum_joint(a), sigma=10)))
#     b_shift = np.argmin(np.gradient(gaussian_filter(calculate_1D_spectrum_joint(b), sigma=10)))
# else:
a_shift = 0
#Free memory
del fft_img

# 0-padding to preserve image size
a_padded = zero_pad_freq(a,SIZE, a_shift)
b_padded = zero_pad_freq(b, SIZE, -a_shift)
#Free memory
del a,b

sections = [a_padded, b_padded]
#Free memory
del a_padded, b_padded

# Apply Hamming window to each section
for section in sections:
    M_section = section.shape[1]
    hamming_win = hamming_window(M_section)
    section = section * hamming_win[np.newaxis, :]


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
plt.savefig('sublookA_'+str(perform_shift)+'.png')

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
plt.savefig('sublookB_'+str(perform_shift)+'.png')

#Display of the reconstructed image
recon_intensity = (np.abs(spatial_sectA)**2 + np.abs(spatial_sectB)**2)/2
plt.figure()
plt.set_cmap('gray')
plt.title('Reconstructed Image')
plt.xlabel('Range')
plt.ylabel('Azimuth')
plt.imshow(data_plot(np.sqrt(recon_intensity), threshold))
plt.savefig('2look_shift_'+str(perform_shift)+'.png')

# Difference Image (for visualizing the reconstruction error)
plt.figure()
ratio = np.divide(recon_intensity + 1e-10, (np.abs(complex_image) ** 2) + 1e-10)
ratio = np.clip(ratio, 0, 4)
plt.imshow(ratio, vmax=4)
plt.title('Difference Image')
plt.colorbar()
plt.savefig('ratio_image_shift_'+str(perform_shift)+'.png')
plt.show()


