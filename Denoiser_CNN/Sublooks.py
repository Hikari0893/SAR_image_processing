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
picture = np.load('/home/tonix/Documents/Dayana/converted_image.npy')


# Reconstruct the complex image from its real and imaginary parts.
complex_image = picture[:, :, 0] + 1j * picture[: ,: , 1]


#Free memory
del picture


(ROWS,COLUMNS) = complex_image.shape
SIZE = (ROWS,COLUMNS)

# Applying FFT by row and FFTshift
fft_img = fftshift(fft(complex_image,axis=1))
np.save('abs_complex.npy', np.abs(complex_image))

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


np.save('spatial_sect1.npy', sections[0])
np.save('spatial_sect2.npy', sections[1])

#Free memory
del sections

# Applying IFFT by row and IFFTshift for each sublook 
spatial_sect1 = ifft(ifftshift(np.load('spatial_sect1.npy')), axis=1)

print(sys.getsizeof(spatial_sect1))
np.save('sublook1.npy', spatial_sect1)

np.save('sublookA.npy', np.stack((np.real(spatial_sect1), np.imag(spatial_sect1)), axis = 2))

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

np.save('sublookB.npy', np.stack((np.real(spatial_sect2), np.imag(spatial_sect2)), axis = 2))

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


#Display of the Original image
plt.figure()
plt.set_cmap('gray')
plt.title('Original Image')
plt.xlabel('Range')
plt.ylabel('Azimuth')
plt.imshow(rebin(np.abs(custom_bytescale(np.load('abs_complex.npy', mmap_mode='r'))), (500, 500)))
plt.savefig("Original_Image.png")
plt.show()

#Free memory
remove("abs_complex.npy")
remove("abs_reconst_img.npy")

remove("difference.npy")
remove("reconstructed.npy")


print("...Code executed...")


