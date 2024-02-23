import numpy as np
from scipy.fft import fft, fftshift, ifft, ifftshift

from cnn_despeckling.utils import hamming_window
from Preprocessing__.Joint_Plots import calculate_1D_spectrum_joint
from cnn_despeckling.Patches import create_patches
from cnn_despeckling.utils import get_ovr, analyse_spectra

rg_sta = 1000
rg_end = 3000
az_sta = 1000
az_end = 3000
crop = [rg_sta, rg_end, az_sta, az_end]

# Testing ids
# id = 'neustrelitz'
# id = 'hamburg'

# Training ids
# id = 'koeln'
# id = 'shenyang'
# id = 'bangkok'
# id = 'enschene'

full_image = True
ids = ['koeln', 'shenyang', 'bangkok', 'enschene']

# rg = 0, az = 1
axis = 0
alpha = 0.6

sub_alpha = 0.6

for id in ids:
    SIZE = []
    picture = np.load('../data/training/tsx_hh_slc_'+id+'.npy')
    # Reconstruct the complex image from its real and imaginary parts.
    if full_image:
        complex_image = picture[:, :, 0] + 1j * picture[: ,: , 1]
    else:
        complex_image = picture[rg_sta:rg_end, az_sta:az_end, 0] + 1j * picture[rg_sta:rg_end, az_sta:az_end, 1]

    if axis == 0:
        # Transpose if range sublooks are desired
        complex_image = complex_image.T

    SIZE = complex_image.shape

    # Applying FFT by row and FFTshift
    fft_img = fftshift(fft(complex_image, axis=1), axes=1)
    start, end, zD = analyse_spectra(complex_image[0:1000, 0:1000])

    start = int(start * SIZE[1])
    end = int(end * SIZE[1])
    zD = int(zD * SIZE[1])

    print("Spectra start/end and zD:")
    print((start, end, zD))

    spectra = fft_img[:, start:end]
    # Inverse of Hamming window
    M = spectra.shape[1]
    inverse_hamming = 1 / hamming_window(M, alpha=alpha)
    spectra = spectra * inverse_hamming[np.newaxis, :]

    M = fft_img.shape[1]
    spectra2 = np.copy(fft_img)
    spectra2[:, start:end] = spectra
    a = np.copy(spectra2)
    b = np.copy(spectra2)

    # Zero padding
    a[:, zD:-1] = 0
    b[:, 0:zD] = 0

    a_padded = np.copy(a)
    b_padded = np.copy(b)

    hamming_win = hamming_window(zD - start, alpha=sub_alpha)
    a_padded[:, start:zD] *= hamming_win

    hamming_win = hamming_window(end - zD, alpha=sub_alpha)
    b_padded[:, zD:end] *= hamming_win

    # # Half spacing test
    # a_padded = a_padded[:, 0:zD]
    # b_padded = b_padded[:, zD:-1]

    print("Done with zero padding and Hamming processing...")

    # Applying IFFT by row and IFFTshift for each sublook
    spatial_sectA = ifft(ifftshift(a_padded, axes=1), axis=1)
    spatial_sectB = ifft(ifftshift(b_padded, axes=1), axis=1)
    print("Done with ifft...")

    # Go back to original axes when processing range sublooks
    if axis == 0:
        spatial_sectA = spatial_sectA.T
        spatial_sectB = spatial_sectB.T
    np.save('../data/training/sublookA_'+id+'.npy', np.stack((np.real(spatial_sectA), np.imag(spatial_sectA)),
                                                                  axis=2))

    np.save('../data/training/sublookB_'+id+'.npy', np.stack((np.real(spatial_sectB), np.imag(spatial_sectB)),
                                                                  axis=2))

    print("Done with id: " + str(id))


    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.imshow(np.abs(spatial_sectA[0:300, 100:330]), vmax=300, cmap="gray")
    # plt.figure()
    # plt.imshow(np.abs(spatial_sectB[0:300, 100:330]), vmax=300, cmap="gray")
    # plt.figure()
    # plt.imshow(np.abs(complex_image[0:300, 100:330]), vmax=300, cmap="gray")
    #
    # plt.plot(calculate_1D_spectrum_joint(a_padded))
    # plt.plot(calculate_1D_spectrum_joint(b_padded))
    # plt.plot(calculate_1D_spectrum_joint(fft_img))

    print("-----")


# print("Creating patches from sublooks...")
# stride = 256
# output_base_folder = '../data/training_patches/'
# A_files = sorted(glob.glob('../data/training/sublookA*'))
# B_files = sorted(glob.glob('../data/training/sublookB*'))
# create_patches([A_files, B_files], stride, output_base_folder)
