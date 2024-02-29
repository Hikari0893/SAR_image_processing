import numpy as np
from scipy.fft import fft, fftshift, ifft, ifftshift

from cnn_despeckling.utils import hamming_window
from Preprocessing__.Joint_Plots import calculate_1D_spectrum_joint
from cnn_despeckling.utils import analyse_spectra
import matplotlib.pyplot as plt

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

# Processing parameters
# For TSX, axis=0 AZIMUTH, axis=1 RANGE (columns = azimuth lines, rows = range lines)
proc_axis = 1
full_image = 1
debug = 0

ids = ['koeln', 'shenyang', 'bangkok', 'enschene', 'neustrelitz']
if proc_axis == 0:
    alpha = 0.61
    sub_alpha = 0.61
    procid = 'az'

else:
    alpha = 0.6
    sub_alpha = 0.6
    procid = 'rg'

for id in ids:
    SIZE = []

    if id == 'neustrelitz':
        picture = np.load('../data/testing/tsx_hh_slc_' + id + '.npy')
    else:
        picture = np.load('../data/training/tsx_hh_slc_'+id+'.npy')

    # Reconstruct the complex image from its real and imaginary parts.
    if full_image:
        complex_image = picture[:, :, 0] + 1j * picture[: ,: , 1]
    else:
        complex_image = picture[rg_sta:rg_end, az_sta:az_end, 0] + 1j * picture[rg_sta:rg_end, az_sta:az_end, 1]

    # Transposing if processing azimuth sublooks...
    if proc_axis == 0:
        complex_image = complex_image.T
        print("Processing AZIMUTH sublooks!")
    else:
        print("Processing RANGE sublooks!")


    SIZE = complex_image.shape
    fft_img = fftshift(fft(complex_image, axis=1), axes=1)
    start, end, zD = analyse_spectra(complex_image[0:1000, 0:1000])

    start = int(start * SIZE[1])
    end = int(end * SIZE[1])
    zD = int(zD * SIZE[1])

    print("Spectra start/end and zD:")
    print((start, end, zD))

    # Shift the spectrum to 0
    hlf = fft_img.shape[1] // 2
    spec_shift = hlf - zD
    fft_img = np.roll(fft_img, spec_shift, axis=1)
    start += spec_shift
    end += spec_shift
    zD += spec_shift

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

    # same OSF as input, have to cut exactly in the middle to avoid sublooks with different sizes
    a_padded = a_padded[:, 0:hlf]
    if np.mod(M, 2) == 0:
        b_padded = b_padded[:, hlf-1:-1]
    else:
        b_padded = b_padded[:, hlf:-1]

    print(f"ZD : {zD}, HLF : {hlf}")

    try:
        a_padded + b_padded
    except ValueError:
        print("Sublooks have different dimensions. Skipping scene...")
        continue
    else:
        print("Sublooks have the same dimensions after zero-Padding")

    print("Done with zero-padding and Hamming processing...")

    # Applying IFFT by row and IFFTshift for each sublook
    spatial_sectA = ifft(ifftshift(a_padded, axes=1), axis=1)
    spatial_sectB = ifft(ifftshift(b_padded, axes=1), axis=1)

    if debug:
        plt.figure()
        plt.plot(calculate_1D_spectrum_joint(fft_img))
        plt.plot(calculate_1D_spectrum_joint(spectra2))
        plt.plot(calculate_1D_spectrum_joint(a_padded))
        plt.plot(calculate_1D_spectrum_joint(b_padded))

        m = end - start
        osf = M / m
        os_pad = int((M - int(M / osf)))

        m_sa = hlf - start
        osf_sa = M / m_sa
        os_pad_sa = int((M - int(M / osf_sa)))
        # Oversampling factor is now twice as large in the sublooks after 2 sublooks

        if proc_axis == 1:
            ccrop = complex_image[400:550, 625:775]
            acrop = spatial_sectA[400:550, 625:775]
            bcrop = spatial_sectB[400:550, 625:775]
            # ccrop = complex_image2[400:550, 1230:1550]

        else:
            ccrop = complex_image[625:775,400:550]
            acrop = spatial_sectA[625:775,400:550]


        # plt.figure()
        # plt.imshow(np.abs(spatial_sectA)[0:500,0:500], vmax=300, cmap="gray")
        # plt.figure()
        # plt.imshow(np.abs(spatial_sectB)[0:500,0:500], vmax=300, cmap="gray")
        # plt.figure()
        # plt.imshow(np.abs(complex_image)[0:500,0:1000], vmax=300, cmap="gray")

        histo=False
        if histo:
            plt.figure()
            plt.hist(np.abs(ccrop).flatten(), bins=100)
            plt.hist(np.abs(acrop).flatten(), bins=100, alpha=0.7)
            plt.gca().legend((f'Amplitude histogram (full res),'
                              f' mean={np.mean(np.abs(ccrop)):.2f}, var={np.var(np.abs(ccrop)):.2f}',
                              f'Amplitude histogram (SL_a, sub_alpha={sub_alpha},'
                              f' mean={np.mean(np.abs(acrop)):.2f}, var={np.var(np.abs(acrop)):.2f}',))

            plt.figure()
            plt.hist(np.real(ccrop).flatten(), bins=100)
            plt.hist(np.real(acrop).flatten(), bins=100, alpha=0.7)

            plt.gca().legend(('Histogram of the I/Q component (full res)',
                              'Histogram of the I/Q component (SL_a, sub_alpha = ' + str(sub_alpha) +')'))

        stop = True

    print("Done with ifft...")

    # Go back to original axes when processing azimuth sublooks
    if proc_axis == 0:
        spatial_sectA = spatial_sectA.T
        spatial_sectB = spatial_sectB.T

    if id == 'neustrelitz':
        np.save(f"../data/testing/{procid}_sublookA_{id}.npy", np.stack((np.real(spatial_sectA), np.imag(spatial_sectA)),
                                                                      axis=2))
        np.save(f"../data/testing/{procid}_sublookB_{id}.npy", np.stack((np.real(spatial_sectB), np.imag(spatial_sectB)),
                                                                      axis=2))
    else:
        np.save(f"../data/training/{procid}_sublookA_{id}.npy", np.stack((np.real(spatial_sectA), np.imag(spatial_sectA)),
                                                                      axis=2))
        np.save(f"../data/training/{procid}_sublookB_{id}.npy", np.stack((np.real(spatial_sectB), np.imag(spatial_sectB)),
                                                                      axis=2))

    if debug:
        arr = np.load(f"../data/fsar/{procid}_sublookA_{id}.npy")
        arr = arr[..., 0] + 1j * arr[..., 1]
        suba = np.abs(arr)

        arr = np.load(f"../data/fsar/{procid}_sublookB_{id}.npy")
        arr = arr[..., 0] + 1j * arr[..., 1]
        subb = np.abs(arr)

        plt.figure()
        plt.imshow(suba, vmax=300, cmap="gray")

        plt.figure()
        plt.imshow(subb, vmax=300, cmap="gray")

        amp_org = np.abs(complex_image)
        if proc_axis == 0:
            amp_org = amp_org.T

        plt.figure()
        plt.imshow(amp_org, vmax=300, cmap="gray")
        plt.show()

        stop=1

    print("Done with id: " + str(id))

    print("-----")