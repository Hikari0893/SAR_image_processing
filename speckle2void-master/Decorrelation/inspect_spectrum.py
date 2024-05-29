import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy import signal

def load_npy(file_name):
    image = np.load(file_name)
    
    # Check if the loaded image has three dimensions
    if image.ndim != 3 or image.shape[2] < 2:
        raise ValueError("Input array must have at least three dimensions with the third dimension having size of at least 2.")
    
    complex_image = image[0:10000,0:10000,0] + 1j * image[0:10000,0:10000,1]
    return complex_image

def inspect_spectrum(input_file):
    complex_SAR = load_npy(input_file)
    print("Complex SAR shape:", complex_SAR.shape)
    
    inphase = complex_SAR.real
    inquad = complex_SAR.imag
    img_complex = inphase.astype(np.float64) + 1j * inquad.astype(np.float64)

    r, c = img_complex.shape

    fC = np.fft.fft2(img_complex)
    S = np.real(fC * np.conj(fC))
    
    temp1 = np.sqrt(np.mean(np.fft.fftshift(S), axis=0))
    temp1 /= np.max(temp1)
    x1 = np.linspace(-1, 1, c)
    plt.figure()
    plt.plot(x1, temp1, 'o')
    plt.title('Spectrum Mean along Axis 0')
    plt.xlabel('Normalized Frequency')
    plt.ylabel('Amplitude')
    

    temp1 = np.sqrt(np.mean(np.fft.fftshift(S), axis=1))
    temp1 /= np.max(temp1)
    x1 = np.linspace(-1, 1, r)
    plt.figure()
    plt.plot(x1, temp1, 'o')
    plt.title('Spectrum Mean along Axis 1')
    plt.xlabel('Normalized Frequency')
    plt.ylabel('Amplitude')
    plt.show()
    
    
def analyse_spectra(arr):
    
    S = np.fft.fft2(arr)
    p = np.zeros((S.shape[1])) # azimut (ncol)
    for i in range(S.shape[1]):
        p[i] = np.mean(np.abs(S[:, i]))
    sp = p[::-1]
    c = np.real(np.fft.ifft(np.fft.fft(p)*np.conjugate(np.fft.fft(sp))))
    d1 = np.unravel_index(c.argmax(),p.shape[0])
    d1 = d1[0]
    shift_az_1 = int(round(-(d1-1)/2))%p.shape[0]+int(p.shape[0]/2)
    p2_1 = np.roll(p,shift_az_1)
    shift_az_2 = int(round(-(d1-1-p.shape[0])/2))%p.shape[0]+int(p.shape[0]/2)
    p2_2 = np.roll(p,shift_az_2)
    window = signal.gaussian(p.shape[0], std=0.2*p.shape[0])
    test_1 = np.sum(window*p2_1)
    test_2 = np.sum(window*p2_2)
    # make sure the spectrum is symetrized and zero-Doppler centered
    if test_1>=test_2:
        shift = shift_az_1/p.shape[0]
    else:
        shift = shift_az_2/p.shape[0]
    p = np.roll(p,int(shift*p.shape[0]),axis=0)
    # Choose a threshold (for example, 10% of the maximum magnitude)
    threshold = 0.07 * np.max(np.abs(gaussian_filter(p, sigma=0.2*p.shape[0])))

    # Find start frequency
    start_freq = None
    for freq_index, magnitude in enumerate(p):
        if magnitude > threshold:
            start_freq = freq_index
            break

    # Find end frequency
    end_freq = None
    for freq_index in range(len(p) - 1, 0, -1):
        magnitude = p[freq_index]
        if magnitude > threshold:
            end_freq = freq_index
            break

    M = arr.shape[1]
    # zD = np.argmax(gaussian_filter(calculate_1D_spectrum_joint(fft_img), sigma=2))
    zD = int((end_freq - start_freq) / 2) + start_freq

    return start_freq/M, end_freq/M, zD/M, shift    


#input_file = '/home/tonix/Documents/Dayana/Dataset/training/TESTING_/slc_005.npy'
##inspect_spectrum(input_file)
#start_freq, end_freq, zD, shift = analyse_spectra(input_file)

## Print the results
#print(f"Start Frequency: {start_freq}")
#print(f"End Frequency: {end_freq}")
#print(f"Zero-Doppler Frequency: {zD}")
#print(f"Shift: {shift}")
