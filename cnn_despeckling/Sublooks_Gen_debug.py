import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift, ifft, ifftshift
from scipy.ndimage import gaussian_filter

from cnn_despeckling.utils import hamming_window, smooth
from zero_pad_freq import zero_pad_freq
from Preprocessing__.Joint_Plots import calculate_1D_spectrum_joint

def get_ovr(arr):
    fft_img = fftshift(fft(arr, axis=1))
    M = fft_img.shape[1]
    # inverse_hamming = 1 / hamming_window(M, alpha=0.6)
    # fft_img = fft_img * inverse_hamming[np.newaxis, :]
    grad = np.gradient(gaussian_filter(calculate_1D_spectrum_joint(fft_img), sigma=6))
    zD = np.argmax(gaussian_filter(calculate_1D_spectrum_joint(fft_img), sigma=0))
    start = np.argmax(grad)
    end = np.argmin(grad)
    return start/M, end/M, zD/M

SIZE = []

def data_plot(im, threshold):
    im = np.clip(im, 0, threshold)
    im = im / threshold * 255
    
    return im

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
axis = 0

ids = ['koeln']
for id in ids:
    SIZE = []
    picture = np.load('../data/training/debug_tsx_hh_slc_'+id+'.npy')
    # Reconstruct the complex image from its real and imaginary parts.
    if full_image:
        complex_image = picture[:, :, 0] + 1j * picture[: ,: , 1]
    else:
        complex_image = picture[rg_sta:rg_end, az_sta:az_end, 0] + 1j * picture[rg_sta:rg_end, az_sta:az_end, 1]

    if axis == 0:
        complex_image = complex_image.T

    SIZE = complex_image.shape
    # Applying FFT by row and FFTshift
    fft_img = fftshift(fft(complex_image, axis=1), axes=1)

    start, end, zD = get_ovr(complex_image[0:1000, 0:1000])
    start = int(start * SIZE[1])
    end = int(end * SIZE[1])
    zD = int(zD * SIZE[1])

    print("Spectra start/end and zD:")
    print((start, end, zD))

    spectra = fft_img[:, start:end]
    # Inverse of Hamming window
    M = spectra.shape[1]
    inverse_hamming = 1 / hamming_window(M, alpha=0.6)
    spectra = spectra * inverse_hamming[np.newaxis, :]

    M = fft_img.shape[1]
    spectra2 = np.copy(fft_img)
    spectra2[:, start:end] = spectra
    a = np.copy(spectra2)
    b = np.copy(spectra2)

    # Zero padding
    a[:, zD:-1] = 0
    b[:, 0:zD] = 0

    for alpha in [0.6, 0.8, 0.95]:

        a_padded = np.copy(a)
        b_padded = np.copy(b)

        hamming_win = hamming_window(zD - start, alpha=alpha)
        a_padded[:, start:zD] *= hamming_win

        hamming_win = hamming_window(end - zD, alpha=alpha)
        b_padded[:, zD:end] *= hamming_win
        print("Done with zero Padding and hamming windows...")

        # Applying IFFT by row and IFFTshift for each sublook
        spatial_sectA = ifft(ifftshift(a_padded, axes=1), axis=1)
        spatial_sectB = ifft(ifftshift(b_padded, axes=1), axis=1)

        plt.plot(calculate_1D_spectrum_joint(fftshift(fft(spatial_sectA, axis=1), axes=1)))
        plt.plot(calculate_1D_spectrum_joint(fftshift(fft(spatial_sectB, axis=1), axes=1)))
        plt.plot(calculate_1D_spectrum_joint(fftshift(fft(complex_image, axis=1), axes=1)))
        plt.show()

        if axis == 0:
            spatial_sectA = spatial_sectA.T
            spatial_sectB = spatial_sectB.T
        np.save('../data/training/debug_sublookA_koeln.npy', np.stack((np.real(spatial_sectA), np.imag(spatial_sectA)),
                                                                 axis = 2))


        np.save('../data/training/debug_sublookB_koeln.npy', np.stack((np.real(spatial_sectB), np.imag(spatial_sectB)),
                                                                 axis = 2))



        print("Done with id: " + str(id))
        print("-----")

        paa = np.load('../data/training/debug_sublookA_koeln.npy')
        pbb = np.load('../data/training/debug_sublookB_koeln.npy')
        slc = np.load('../data/training/debug_tsx_hh_slc_koeln.npy')



        # ca = paa[0:500, 1500:2000, 0] + 1j * paa[0:500, 1500:2000, 1]
        # cb = pbb[0:500, 1500:2000, 0] + 1j * pbb[0:500, 1500:2000, 1]
        # cs = slc[0:500, 1500:2000, 0] + 1j * slc[0:500, 1500:2000, 1]
        ca = paa[..., 0] + 1j * paa[..., 1]
        cb = pbb[..., 0] + 1j * pbb[..., 1]
        cs = slc[..., 0] + 1j * slc[..., 1]

        if axis == 0:
            ca = ca.T
            cb = cb.T
            cs = cs.T

        plt.figure();
        plt.imshow(np.abs(ca), vmax=500, cmap="gray")
        plt.figure();
        plt.imshow(np.abs(cb), vmax=500, cmap="gray")
        plt.figure();
        plt.imshow(np.abs(cs), vmax=500, cmap="gray")

        win = 7
        array = [ca, cb]
        coh = np.abs(smooth(array[0] * np.conj(array[1]), win)
                     / np.sqrt(smooth(array[0] * np.conj(array[0]), win)
                               * smooth(array[1] * np.conj(array[1]), win)))

        # plt.figure()
        # plt.imshow(coh, vmax=1, cmap="gray")
        print("Coherence mean: " +str(np.mean(coh)))

        plt.figure()
        plt.plot(calculate_1D_spectrum_joint(fftshift(fft(ca, axis=1), axes=1)))
        plt.plot(calculate_1D_spectrum_joint(fftshift(fft(cb, axis=1), axes=1)))
        plt.plot(calculate_1D_spectrum_joint(fftshift(fft(cs, axis=1), axes=1)))
        plt.show()

        dsada = 1