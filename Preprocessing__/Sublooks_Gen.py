import sys
from os import remove
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift, ifft, ifftshift
from scipy.ndimage import gaussian_filter

from hamming_window import hamming_window
from zero_pad_freq import zero_pad_freq
from custom_bytescale import custom_bytescale
from rebin import rebin
from Joint_Plots import calculate_1D_spectrum_joint
from Hist import display_histogram
from display_1D_spectrum_norm import display_1D_spectrum_norm

def get_ovr(arr, axis=1):
    fft_img = fftshift(fft(arr, axis=axis))
    M = fft_img.shape[1]
    inverse_hamming = 1 / hamming_window(M)
    fft_img = fft_img * inverse_hamming[np.newaxis, :]
    shift = np.argmax(np.gradient(gaussian_filter(calculate_1D_spectrum_joint(fft_img), sigma=10)))
    return shift/M

# Loading image
picture = np.load('../data/training/tsx_hh_slc_crop.npy')
# rows = range lines (axis0), columns = azimuth lines (axis1)

rg_sta = 1000
rg_end = 4000
az_sta = 1000
az_end = 4000
# Reconstruct the complex image from its real and imaginary parts.

# amp = abs(picture[rg_sta:rg_end, az_sta:az_end, 0] + 1j * picture[rg_sta:rg_end, az_sta:az_end, 1])
# plt.imshow(amp, vmax=800)

debug = True
full_image = True
if full_image:
    complex_image = picture[:, :, 0] + 1j * picture[: ,: , 1]
else:
    complex_image = picture[rg_sta:rg_end, az_sta:az_end, 0] + 1j * picture[rg_sta:rg_end, az_sta:az_end, 1]

intermediate_spectra = []

#Free memory
del picture


(ROWS,COLUMNS) = complex_image.shape
SIZE = (ROWS,COLUMNS)

# Applying FFT in either range (frequency domain) or azimuth (Doppler frequency domain) and FFTshift
fft_img = fftshift(fft(complex_image,axis=1))

# Returns shift multiplier
shift = get_ovr(complex_image[0:1000,0:1000])
pix_shift = int(shift * SIZE[1])

fft_img = fft_img[:,pix_shift:SIZE[1]-pix_shift]
np.save('abs_complex.npy', np.abs(complex_image))

# Display the 1D spectrum
intermediate_spectra.append(calculate_1D_spectrum_joint(fft_img))
if debug:
    display_1D_spectrum_norm(fft_img, 'Average spectrum across azimuth')
    # display_histogram(fft_img, 'Average spectrum across azimuth')

#Free memory
del complex_image


# Inverse of Hamming window
M = fft_img.shape[1]
inverse_hamming = 1 / hamming_window(M)
fft_img = fft_img * inverse_hamming[np.newaxis, :]

if debug:
    # display_histogram(fft_img, 'After Inverse Hamming')
    #Display spectrum in 1D after de-Hamming applied
    display_1D_spectrum_norm(fft_img, 'After Inverse Hamming')

intermediate_spectra.append(calculate_1D_spectrum_joint(fft_img))

# Change the overlap calculation and segmentation for n sublooks
# Snumber_SL = 2  # number of sublooks, change as desired
overlap_factor = 0  # for 30% overlap
c = int((overlap_factor * M)/2)
a = fft_img[:,0:M//2+c]
b = fft_img[:,M//2-c:M]

if debug:
    #Display spectrum in 1D for each sublook
    display_1D_spectrum_norm(a,'Sublook A Before Hamming')
    display_1D_spectrum_norm(b, 'Sublook B Before Hamming')

intermediate_spectra.append(calculate_1D_spectrum_joint(a))
intermediate_spectra.append(calculate_1D_spectrum_joint(b))

# display_histogram(a, 'Sublook A Before Hamming')
# display_histogram(b, 'Sublook B Before Hamming')

#Free memory
del fft_img

sections = [a, b]
padded = []
# Apply Hamming window to each section
for section in sections:
    M_section = section.shape[1]
    hamming_win = hamming_window(M_section)
    # for col in range(section.shape[0]):
    #     section[col] *= hamming_win
    padded.append(section * hamming_win[np.newaxis, :])

# 0-padding to preserve image size
a_padded = zero_pad_freq(padded[0], SIZE)
b_padded = zero_pad_freq(padded[1], SIZE)

sections = [a_padded, b_padded]
if debug:
    # Display spectrum for each sublook after 0-padding
    display_1D_spectrum_norm(a_padded, 'Sublook A 0-padding')
    display_1D_spectrum_norm(b_padded, 'Sublook B 0-padding')
    # display_histogram(a_padded, 'Sublook A 0-padding')
    # display_histogram(b_padded, 'Sublook B 0-padding')

intermediate_spectra.append(calculate_1D_spectrum_joint(a_padded))
intermediate_spectra.append(calculate_1D_spectrum_joint(b_padded))

np.save('spatial_sect1.npy', sections[0])
np.save('spatial_sect2.npy', sections[1])

#Free memory
del sections

if debug:
    # Display spectrum for each sublook after Hamming
    display_1D_spectrum_norm(np.load('spatial_sect1.npy', mmap_mode= 'r'), 'Sublook A_padd After Hamming')
    display_1D_spectrum_norm(np.load('spatial_sect2.npy', mmap_mode= 'r'), 'Sublook A_padd After Hamming')
    # display_histogram(np.load('spatial_sect1.npy', mmap_mode= 'r'), 'Sublook A_padd After Hamming')
    # display_histogram(np.load('spatial_sect2.npy', mmap_mode= 'r'), 'Sublook B_padd After Hamming')

intermediate_spectra.append(calculate_1D_spectrum_joint(np.load('spatial_sect1.npy', mmap_mode= 'r')))
intermediate_spectra.append(calculate_1D_spectrum_joint(np.load('spatial_sect2.npy', mmap_mode= 'r')))

if debug:
    # Spectra at each preprocessing step
    plt.figure(figsize=(12, 7))
    steps = ["1: FFT", "2: After Inverse Hamming", "3: Sublook A", "4: Sublook B ","5: Sublook A 0-Padding", "6: Sublook B 0-Padding", "7: Sublook A_padd After Hamming", "8: Sublook B_padd After Hamming"]

    for idx, spectrum in enumerate(intermediate_spectra):
        plt.plot(spectrum, label=steps[idx])

    plt.legend()
    plt.title('Comparison of 1D Magnitude Spectra across Steps')
    plt.xlabel('Frequency Index')
    plt.ylabel('Magnitude')
    plt.grid(True)
    #plt.savefig('All_in.png')
    plt.show()

# plt.figure()
# plt.plot(intermediate_spectra[4])
# plt.plot(intermediate_spectra[5])
# plt.show()
#
# plt.figure()
# plt.plot(intermediate_spectra[6])
# plt.plot(intermediate_spectra[7])
# plt.show()

# Applying IFFT by row and IFFTshift for each sublook 
spatial_sect1 = ifft(ifftshift(np.load('spatial_sect1.npy')), axis=1)

print(sys.getsizeof(spatial_sect1))
np.save('sublook1.npy', spatial_sect1)

#Free memory
del spatial_sect1
remove ("spatial_sect1.npy")

if debug:
    #Sublook display
    plt.figure()
    plt.set_cmap('gray')
    plt.title('Sublook A')
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    amp_a = np.abs(np.load('sublook1.npy', mmap_mode='r'))
    vmax_amp = 3*np.mean(amp_a)
    plt.imshow(rebin(amp_a, (500, 500)), vmax=vmax_amp)
    plt.savefig("Sublook_A.png")
    plt.show()

#IFFT by row and IFFTshift
spatial_sect2 = ifft(ifftshift(np.load('spatial_sect2.npy',  mmap_mode= 'r')), axis=1)

print(sys.getsizeof(spatial_sect2))
np.save('sublook2.npy', spatial_sect2)

if debug:
    #Sublook display
    plt.figure()
    plt.set_cmap('gray')
    plt.title('Sublook B')
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    amp_b = np.abs(np.load('sublook2.npy', mmap_mode='r'))
    plt.imshow(rebin(amp_b, (500, 500)), vmax=vmax_amp)
    plt.savefig("Sublook_B.png")
    plt.show()


#Free memory
remove ("spatial_sect2.npy")
del spatial_sect2

# Image is not reconstructed coherently, i.e. its always an incoherent reconstruction!!
# Reconstruct image from the two sublooks
# np.save('reconstructed.npy',((np.load('sublook1.npy',  mmap_mode= 'r') + np.load('sublook2.npy',  mmap_mode= 'r'))/2))

if debug:
    # We save it as intensity
    reconstructed_intensity = (amp_a**2 + amp_b**2)/2
    np.save('reconstructed.npy', reconstructed_intensity)
    #Display of the reconstructed image
    plt.figure()
    plt.set_cmap('gray')
    plt.title('Reconstructed Image')
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    plt.imshow(rebin(np.sqrt(reconstructed_intensity), (500, 500)), vmax=vmax_amp)
    plt.savefig("Reconstructed_Image.png")
    plt.show()

    # Display Original Image
    orig_amp = np.abs(np.load('abs_complex.npy', mmap_mode='r'))
    plt.subplot(1, 3, 1)
    plt.imshow(rebin(orig_amp, (500,500)), cmap='gray', aspect='auto', vmax=vmax_amp)
    plt.title('Original Image')
    plt.colorbar()

    # Display Reconstructed Image
    plt.subplot(1, 3, 2)
    plt.imshow(rebin(np.sqrt(reconstructed_intensity), (500,500)), cmap='gray', aspect='auto', vmax=vmax_amp)
    plt.title('Reconstructed Image')
    plt.colorbar()

    # Difference Image (for visualizing the reconstruction error)
    plt.subplot(1, 3, 3)
    ratio = np.divide(reconstructed_intensity + 1e-10, (orig_amp**2) + 1e-10)
    ratio = np.clip(ratio, 0, 4)
    plt.imshow(rebin(ratio, (500,500)), cmap='gray', aspect='auto', vmax=4)
    plt.title('Difference Image')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig('reconstructed_image.png')
    plt.show()

# Creating histogram
#difference_flatten = (np.load('difference.npy', mmap_mode='r')).flatten()

if debug:
    plt.figure(figsize=(10, 7))
    plt.hist((np.load('abs_complex.npy')).flatten(), bins=50, alpha=0.5, label='Original Image')
    plt.hist((np.abs(np.load('reconstructed.npy'))).flatten(), bins=50, alpha=0.5, label='Reconstructed Image')
    #plt.hist(difference_flatten, bins=50, alpha=0.5, label='Difference Image')

    plt.title('Distribution of Pixel Intensities')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Distribution')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig('Histogram_SL.png')
    plt.show()

print("I am here")


