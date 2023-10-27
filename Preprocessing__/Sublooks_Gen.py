import sys
from os import remove
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift, ifft, ifftshift



from hamming_window import hamming_window
from zero_pad_freq import zero_pad_freq
from custom_bytescale import custom_bytescale
from rebin import rebin
from Joint_Plots import calculate_1D_spectrum_joint
from Hist import display_histogram
from display_1D_spectrum_norm import display_1D_spectrum_norm

# Loading image
picture = np.load('converted_image.npy')


# Reconstruct the complex image from its real and imaginary parts.
complex_image = picture[:, :, 0] + 1j * picture[: ,: , 1]

intermediate_spectra = []

#Free memory
del picture


(ROWS,COLUMNS) = complex_image.shape
SIZE = (ROWS,COLUMNS)

# Applying FFT by row and FFTshift
fft_img = fftshift(fft(complex_image,axis=1))
np.save('abs_complex.npy', np.abs(complex_image))

# Display the 1D spectrum
intermediate_spectra.append(calculate_1D_spectrum_joint(fft_img))
display_1D_spectrum_norm(fft_img, 'Average spectrum across columns')
display_histogram(fft_img, 'Average spectrum across columns')

#Free memory
del complex_image


# Inverse of Hamming window
M = fft_img.shape[1]
inverse_hamming = 1 / hamming_window(M)
for col in range(fft_img.shape[0]):
    fft_img[col] *= inverse_hamming

#Display spectrum in 1D after de-Hamming applied
display_1D_spectrum_norm(fft_img, 'After Inverse Hamming')
display_histogram(fft_img, 'After Inverse Hamming')
intermediate_spectra.append(calculate_1D_spectrum_joint(fft_img))


# Change the overlap calculation and segmentation for n sublooks
# Snumber_SL = 2  # number of sublooks, change as desired
overlap_factor = 0  # for 30% overlap
c = int((overlap_factor * M)/2)
a = fft_img[:,0:M//2+c]
b = fft_img[:,M//2-c:M]

#Display spectrum in 1D for each sublook
display_1D_spectrum_norm(a,'Sublook A Before Hamming')
display_1D_spectrum_norm(b, 'Sublook B Before Hamming')

intermediate_spectra.append(calculate_1D_spectrum_joint(a))
intermediate_spectra.append(calculate_1D_spectrum_joint(b))

display_histogram(a, 'Sublook A Before Hamming')
display_histogram(b, 'Sublook B Before Hamming')

#Free memory
del fft_img

# 0-padding to preserve image size
a_padded = zero_pad_freq(a,SIZE)
b_padded = zero_pad_freq(b, SIZE)

# Display spectrum for each sublook after 0-padding
display_1D_spectrum_norm(a_padded, 'Sublook A 0-padding')
display_1D_spectrum_norm(b_padded, 'Sublook B 0-padding')
display_histogram(a_padded, 'Sublook A 0-padding')
display_histogram(b_padded, 'Sublook B 0-padding')

intermediate_spectra.append(calculate_1D_spectrum_joint(a_padded))
intermediate_spectra.append(calculate_1D_spectrum_joint(b_padded))


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


np.save('spatial_sect1.npy', sections[0])
np.save('spatial_sect2.npy', sections[1])

#Free memory
del sections

# Display spectrum for each sublook after Hamming
display_1D_spectrum_norm(np.load('spatial_sect1.npy', mmap_mode= 'r'), 'Sublook A_padd After Hamming')
display_1D_spectrum_norm(np.load('spatial_sect2.npy', mmap_mode= 'r'), 'Sublook A_padd After Hamming')
display_histogram(np.load('spatial_sect1.npy', mmap_mode= 'r'), 'Sublook A_padd After Hamming')
display_histogram(np.load('spatial_sect2.npy', mmap_mode= 'r'), 'Sublook B_padd After Hamming') 

intermediate_spectra.append(calculate_1D_spectrum_joint(np.load('spatial_sect1.npy', mmap_mode= 'r')))
intermediate_spectra.append(calculate_1D_spectrum_joint(np.load('spatial_sect2.npy', mmap_mode= 'r')))

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




# Applying IFFT by row and IFFTshift for each sublook 
spatial_sect1 = ifft(ifftshift(np.load('spatial_sect1.npy')), axis=1)

print(sys.getsizeof(spatial_sect1))
np.save('sublook1.npy', spatial_sect1)

#Free memory
del spatial_sect1
remove ("spatial_sect1.npy")

#Sublook display
plt.figure()
plt.set_cmap('gray')
plt.title('Sublook A')
plt.xlabel('Range')
plt.ylabel('Azimuth')
plt.imshow(rebin(np.abs(custom_bytescale(np.abs(np.load('sublook1.npy', mmap_mode='r')))), (500, 500)))
plt.savefig("Sublook_A.png")
plt.show()

#IFFT by row and IFFTshift
spatial_sect2 = ifft(ifftshift(np.load('spatial_sect2.npy',  mmap_mode= 'r')), axis=1)

print(sys.getsizeof(spatial_sect2))
np.save('sublook2.npy', spatial_sect2)

#Sublook display
plt.figure()
plt.set_cmap('gray')
plt.title('Sublook B')
plt.xlabel('Range')
plt.ylabel('Azimuth')
plt.imshow(rebin(np.abs(custom_bytescale(np.abs(np.load('sublook2.npy', mmap_mode='r')))), (500, 500)))
plt.savefig("Sublook_B.png")
plt.show()


#Free memory
remove ("spatial_sect2.npy")
del spatial_sect2

#Reconstruct image from the two sublooks
np.save('reconstructed.npy',((np.load('sublook1.npy',  mmap_mode= 'r') + np.load('sublook2.npy',  mmap_mode= 'r'))/2))

#Free memory
remove("sublook1.npy")
remove("sublook2.npy")

np.save('abs_reconst_img.npy',np.abs(np.load('reconstructed.npy', mmap_mode='r')) )

#Display of the reconstructed image
plt.figure()
plt.set_cmap('gray')
plt.title('Reconstructed Image')
plt.xlabel('Range')
plt.ylabel('Azimuth')
plt.imshow(rebin(np.abs(custom_bytescale(np.load('abs_reconst_img.npy', mmap_mode='r'))), (500, 500)))
plt.savefig("Reconstructed_Image.png")
plt.show()

# Display Original Image
plt.subplot(1, 3, 1)
plt.imshow(rebin(np.abs(custom_bytescale(np.load('abs_complex.npy', mmap_mode='r'))), (500,500)), cmap='gray', aspect='auto')
plt.title('Original Image')
plt.colorbar()

# Display Reconstructed Image
plt.subplot(1, 3, 2)
plt.imshow(rebin(np.abs(custom_bytescale(np.load('abs_reconst_img.npy', mmap_mode='r'))), (500,500)), cmap='gray', aspect='auto')
plt.title('Reconstructed Image')
plt.colorbar()

# Difference Image (for visualizing the reconstruction error)
plt.subplot(1, 3, 3)
np.save('difference.npy', np.load('abs_reconst_img.npy', mmap_mode='r') - np.load('abs_complex.npy', mmap_mode='r'))
plt.imshow(rebin(np.abs(custom_bytescale(np.load('difference.npy', mmap_mode='r'))), (500,500)), cmap='gray', aspect='auto')
plt.title('Difference Image')
plt.colorbar()

plt.tight_layout()
plt.savefig('reconstructed_image.png')
plt.show()


# Creating histogram
#difference_flatten = (np.load('difference.npy', mmap_mode='r')).flatten()

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

#Free memory
remove("abs_complex.npy")
remove("abs_reconst_img.npy")

remove("difference.npy")
remove("reconstructed.npy")


print("I am here")


