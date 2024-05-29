import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from equalizeIQ import equalizeIQ
from inspect_spectrum import load_npy, analyse_spectra


def decorrelate(input_file, output_file, cf):
    complex_SAR = load_npy(input_file)
    print("Complex SAR shape:", complex_SAR.shape)
    
    inphase = complex_SAR.real
    inquad = complex_SAR.imag
    img_complex = inphase.astype(np.float64) + 1j * inquad.astype(np.float64)

    r, c = img_complex.shape
    intensity_img = np.abs(img_complex) ** 2
    threshold = cf * np.median(intensity_img)

    index_nonpoints = intensity_img < threshold
    index_points = intensity_img >= threshold

    n_nonpoints = np.sum(index_nonpoints)
    n_points = np.sum(index_points)
    var = np.sum(intensity_img[index_nonpoints]) / n_nonpoints

    # Replace the point targets with complex values drawn from a complex circular symmetric Gaussian distribution
    new_points = np.sqrt(var / 2) * (np.random.randn(n_points) + 1j * np.random.randn(n_points))

    img_complex_new = np.copy(complex_SAR)
    img_complex_new[index_points] = new_points
    start_freq, end_freq, zD, shift = analyse_spectra(complex_SAR)
    m_x = end_freq - start_freq
    f_x = shift
    start_freq, end_freq, zD, shift = analyse_spectra(complex_SAR.T)
    m_y = end_freq - start_freq
    f_y = shift

    cout, W = equalizeIQ(img_complex_new, m_x, m_y, f_x, f_y)

    # Plot the frequency spectrum of the decorrelated complex SAR image
    fC = np.fft.fft2(cout)
    S = np.real(fC * np.conj(fC))
    
    temp1 = np.sqrt(np.mean(np.fft.fftshift(S), axis=0))
    temp1 = temp1 / np.max(temp1)
    x1 = np.linspace(-1, 1, c)
    plt.figure()
    plt.plot(x1, temp1, 'o')
    plt.show()

    temp1 = np.sqrt(np.mean(np.fft.fftshift(S), axis=1))
    temp1 = temp1 / np.max(temp1)
    x1 = np.linspace(-1, 1, r)
    plt.figure()
    plt.plot(x1, temp1, 'o')
    plt.show()

    # Replace back the point targets
    cout[index_points] = img_complex[index_points]
    

    #sio.savemat(output_file, {'cout': cout})
    np.save(output_file, cout)


# Example usage:
input_file = '/home/tonix/Documents/Dayana/Dataset/training/TESTING_/slc_005.npy'
output_file = '/home/tonix/Documents/Dayana/speckle2void-master/test_examples/decorr_complex_tsx_SLC_1.npy'
cf = 1
decorrelate(input_file, output_file,  cf)
