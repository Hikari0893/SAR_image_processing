from scipy import special
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import filters
from scipy.ndimage import gaussian_filter
from Preprocessing__.Joint_Plots import calculate_1D_spectrum_joint
from scipy.fft import fft, fftshift, ifft, ifftshift
import torch
from cnn_despeckling.Jarvis import *
from scipy.fft import fft, fftshift, ifft, ifftshift
from scipy import signal

def get_ovr(arr, fsar=False, sigma=6):
    if fsar:
        fft_img = fft(arr, axis=1)
    else:
        fft_img = fftshift(fft(arr, axis=1), axes=1)

    M = fft_img.shape[1]
    grad = np.gradient(gaussian_filter(calculate_1D_spectrum_joint(fft_img), sigma=sigma))
    zD = np.argmax(gaussian_filter(calculate_1D_spectrum_joint(fft_img), sigma=0))
    # grad2 = np.gradient(grad)
    start = np.argmax(grad)
    end = np.argmin(grad)
    return start/M, end/M, zD/M

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

def sublook_shift(fft_img, shift):
    return np.roll(fft_img, int(shift * fft_img.shape[1]), axis=1)

def normalize(batch, tsx=True):
    # batch[batch==0]=1e-2
    if tsx:
        # TSX Params
        L = 1
        M = 10.089038980848645
        m = -1.429329123112601
        c = (special.psi(L) - np.log(L))
        cons = 1e-2
    else:
        # FSAR params
        L = 1
        M = 1.3072924
        m = -8.0590475
        c = (special.psi(L) - np.log(L))
        cons = 1e-8
    return (np.log(batch + cons) - 2 * m) / (2 * (M - m))

def denormalize(batch, tsx=True):
    if tsx:
        # TSX Params
        L = 1
        M = 10.089038980848645
        m = -1.429329123112601
        c = (special.psi(L) - np.log(L))
        cons = 1e-2
    else:
        # FSAR params
        L = 1
        M = 1.3072924
        m = -8.0590475
        c = (special.psi(L) - np.log(L)) * 1
        cons = 1e-8
    return np.exp(2 * np.clip(np.squeeze(batch), 0, 1) * (M - m) + 2*m - c) + cons
    # return np.exp(2 * np.squeeze(batch) * (M - m) + 2*m - c) + cons

def split_sublooks(arr, axis=0, alpha=0.6, debug = False, crop = None):
    sub_alpha = alpha
    # Transposing if processing azimuth sublooks...
    if axis == 0:
        complex_image = arr.T
        print("Processing AZIMUTH sublooks!")
    else:
        complex_image = arr
        print("Processing RANGE sublooks!")

    SIZE = complex_image.shape
    print(f"Processed scene size: {SIZE}")

    # Returns shift multiplier
    start, end, zD, shift = analyse_spectra(complex_image[0:3000, 0:3000])
    fft_img = fft(complex_image, axis=1)
    fft_img = sublook_shift(fft_img, shift)

    start = int(start*SIZE[1])
    end = int(end*SIZE[1])
    zD = int(zD*SIZE[1])

    if debug:
        print("Spectra start/end and zD:")
        print((start,end, zD))

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

    if debug:
        print(f"ZD : {zD}, HLF : {hlf}")

    try:
        a_padded + b_padded
    except ValueError:
        print("Sublooks have different dimensions. Skipping scene...")
    else:
        print("Sublooks have the same dimensions after zero-Padding")

    spatial_sectA = ifft(sublook_shift(a_padded, shift), axis=1)
    spatial_sectB = ifft(sublook_shift(b_padded, shift), axis=1)

    if crop is not None:
        tokeep = int(spatial_sectA.shape[1] * (crop / 100))
        spatial_sectA = spatial_sectA[:, tokeep:-tokeep]
        spatial_sectB = spatial_sectB[:, tokeep:-tokeep]

    if debug:
        plt.figure()
        plt.plot(calculate_1D_spectrum_joint(fft_img))
        plt.plot(calculate_1D_spectrum_joint(spectra2))
        plt.plot(calculate_1D_spectrum_joint(a_padded))
        plt.plot(calculate_1D_spectrum_joint(b_padded))

        m = end - start
        osf = M / m
        os_pad = int((M - int(M / osf)))

        m_sa = zD - start
        osf_sa = M / m_sa
        os_pad_sa = int((M - int(M / osf_sa)))
        # Oversampling factor is now twice as large in the sublooks after 2 sublooks

        if axis == 1:
            ccrop = spatial_sectA[400:550, 625:775]
            acrop = spatial_sectB[400:550, 625:775]
        else:
            ccrop = spatial_sectA[625:775, 400:550]
            acrop = spatial_sectB[625:775, 400:550]

        # plt.figure()
        # plt.imshow(np.abs(ccrop), vmax=1, cmap="gray")
        # plt.figure()
        # plt.imshow(np.abs(acrop), vmax=1, cmap="gray")

        # plt.figure()
        # plt.hist(np.abs(ccrop).flatten(), bins=100)
        # plt.hist(np.abs(acrop).flatten(), bins=100, alpha=0.7)
        # plt.gca().legend((f'Amplitude histogram (full res),'
        #                   f' mean={np.mean(np.abs(ccrop)):.2f}, var={np.var(np.abs(ccrop)):.2f}',
        #                   f'Amplitude histogram (SL_a, sub_alpha={sub_alpha},'
        #                   f' mean={np.mean(np.abs(acrop)):.2f}, var={np.var(np.abs(acrop)):.2f}',))
        #
        # plt.figure()
        # plt.hist(np.real(ccrop).flatten(), bins=100)
        # plt.hist(np.real(acrop).flatten(), bins=100, alpha=0.7)
        #
        # plt.gca().legend(('Histogram of the I/Q component (full res)',
        #                   'Histogram of the I/Q component (SL_a, sub_alpha = ' + str(sub_alpha) + ')'))

        plt.show()

    # Go back to original axes when processing azimuth sublooks
    if axis == 0:
        spatial_sectA = spatial_sectA.T
        spatial_sectB = spatial_sectB.T

    return spatial_sectA, spatial_sectB

def plot_sublooks(pathA, pathB, complex_image):
    arr = np.load(pathA)
    arr = arr[..., 0] + 1j * arr[..., 1]
    suba = np.abs(arr)

    arr = np.load(pathB)
    arr = arr[..., 0] + 1j * arr[..., 1]
    subb = np.abs(arr)

    vmax = np.mean(np.abs(complex_image)) + 3*np.std(np.abs(complex_image))

    plt.figure()
    plt.imshow(suba, vmax=vmax, cmap="gray")

    plt.figure()
    plt.imshow(subb, vmax=vmax, cmap="gray")

    plt.figure()
    plt.imshow(np.abs(complex_image), vmax=vmax, cmap="gray")
    plt.show()
    return

def process_patches_with_model(patch_folder, model, device, desc='patches SLA', patch_index=None):
    processed_patches = []

    # Get all .npy files and sort them
    patch_files = [f for f in os.listdir(patch_folder) if f.endswith('.npy')]
    patch_files.sort(key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))  # Sorting by index

    if patch_index is not None:
        patch_files = [patch_files[patch_index]]

    for filename in tqdm(patch_files, desc='Processing patches ...'):
        # Load patch and convert to PyTorch tensor
        patch_path = os.path.join(patch_folder, filename)
        patch = np.load(patch_path)
        patch = normalize(patch)

        patch_tensor = torch.from_numpy(patch).unsqueeze(0)  # Add batch dimension

        # Move tensor to GPU if available
        patch_tensor = patch_tensor.to(device)
        # Process with the model
        with torch.no_grad():
            model.eval()
            output_tensor = model(patch_tensor)

        # Convert output tensor to NumPy array and move to CPU
        output_array = output_tensor.squeeze(0).cpu().numpy()  # Remove batch dimension
        output_array = denormalize(output_array)
        processed_patches.append(output_array)

    return processed_patches

def mem_process_patches_with_model(patch_list, model, device, tsx=True):
    processed_patches = []
    for patch in tqdm(patch_list, desc='Processing patches ...'):
        if tsx:
            patch[patch==0] += 1e-1

        patch = normalize(patch, tsx=tsx)
        patch_tensor = torch.from_numpy(patch).unsqueeze(0)  # Add batch dimension

        # plt.imshow(np.sqrt(patch_list[0][0, ...]), vmax=300)
        # plt.figure()
        # plt.imshow(patch[0, ...])
        # plt.show()

        # Move tensor to GPU if available
        patch_tensor = patch_tensor.to(device)
        # Process with the model
        with torch.no_grad():
            model.eval()
            output_tensor = model(patch_tensor)

        # Convert output tensor to NumPy array and move to CPU
        output_array = output_tensor.squeeze(0).cpu().numpy()  # Remove batch dimension
        output_array = denormalize(output_array, tsx=tsx)
        processed_patches.append(output_array)

    return processed_patches

def reconstruct_image_from_processed_patches(processed_patches, original_dims, patch_size=256, stride=64, desc='patches SLA'):
    (original_height, original_width) = original_dims
    reconstructed = np.zeros((original_height, original_width), dtype=np.float32)
    counts = np.zeros_like(reconstructed, dtype=np.int32)

    current_x, current_y = 0, 0
    for patch in tqdm(processed_patches, desc=desc):
        # Ensure patch is squeezed from any extra dimensions
        patch = np.squeeze(patch)

        # Place the patch in the reconstructed image
        reconstructed[current_x:current_x + patch_size, current_y:current_y + patch_size] += patch
        counts[current_x:current_x + patch_size, current_y:current_y + patch_size] += 1

        # plt.imshow(np.sqrt(np.abs(reconstructed)), cmap="gray", vmax=400)
        # plt.show()
        # a=2
        # Update coordinates for the next patch
        current_y += stride
        if current_y + patch_size > original_width:
            current_y = 0
            current_x += stride
            if current_x + patch_size > original_height:
                break  # Finished processing all patches

    # Normalize to average overlapping regions
    reconstructed /= np.maximum(counts, 1)  # Avoid division by zero
    return np.abs(reconstructed)

def store_data_and_plot(im, threshold, filename):
    im = np.clip(im, 0, threshold)
    im = im / threshold * 255
    im = Image.fromarray(im.astype(np.uint8)).convert('L')
    im.save(filename + '.png')

def calculate_enl(image, mask=None):
    """
   Calculates the Equivalent Number of Views (ENL) for a SAR image.

    Args:
    - image: An array of NumPy representing the SAR image.
    - mask: A NumPy boolean array of the same size as `image`, where True indicates the pixels belonging to the homogeneous region.
            the pixels belonging to the homogeneous region.

    Returns:
    - ENL: The calculated ENL value for the specified region.
    """
    if mask is None:
        # If no mask is provided, the entire image is used.
        region = image.flatten()
    else:
        # Selects only the pixels of the homogeneous region.
        region = image[mask]

    # Calculate the average and variance of the homogeneous region.
    mean = np.mean(region)
    variance = np.var(region)

    # Calculate the ENL.
    enl = (mean ** 2) / variance

    return enl

def create_patches(arr, stride=256, pat_size=256):
    patch_list = []
    arr = arr.astype(np.float32)
    arr_reshaped = arr.reshape(arr.shape[0], arr.shape[1], arr.shape[2])
    im_h, im_w = arr.shape[1:]
    x_range = range(0, max(im_h - pat_size, 1), stride)
    y_range = range(0, max(im_w - pat_size, 1), stride)

    for x in x_range:
        for y in y_range:
            patch = arr_reshaped[:, x:x + pat_size, y:y + pat_size]
            patch_list.append(patch)
    return patch_list


def create_patches_n(arr, pat_size=256, ovr=0):
    arr = arr.astype(np.float32)
    n = pat_size
    ovr = int(ovr)
    assert n > ovr >= 0
    s1, s2 = np.shape(arr[0,...])
    r1, r2 = (s1 - n) // (n - ovr) + 1, (s2 - n) // (n - ovr) + 1  # number of patches per column/row
    patch_list = []
    for i in range(r1):
        for j in range(r2):
            patch = arr[:, i * (n - ovr):i * (n - ovr) + n, j * (n - ovr):j * (n - ovr) + n]
            patch_list.append(patch)
    return patch_list, r1, r2


def assemble_patches(L_patches, r1, r2, ovr, TEST=False, gaussian_std=0.25):
    """
    :param r1: # number of patches per row
    :param r2: # number of patches per column
    :param L_patches: list of ordered patches
    :param ovr: overlaping pixels (integer)
    :param TEST: To plot more things
    :param gaussian_std: Standard deviation of the 2D gaussian weights window
    :return: reconstructed array
    """

    n = np.shape(L_patches[0])[0]
    ovr = int(ovr)
    assert n > ovr >= 0
    s1, s2 = int((r1 - 1) * (n - ovr) + n), int((r2 - 1) * (n - ovr) + n)
    R = np.zeros((s1, s2))
    R_weights = np.zeros((s1, s2))
    Window = gaussian_window_2D(n, sigma=gaussian_std)  # np.ones((n,n))
    if TEST:
        R_occ = np.zeros((s1, s2));
        Window_occ = np.ones((n, n))
    itr = 0
    for i in range(r1):
        for j in range(r2):
            R[i * (n - ovr):i * (n - ovr) + n, j * (n - ovr):j * (n - ovr) + n] += np.multiply(L_patches[itr], Window)
            R_weights[i * (n - ovr):i * (n - ovr) + n, j * (n - ovr):j * (n - ovr) + n] += Window
            if TEST:
                R_occ[i * (n - ovr):i * (n - ovr) + n, j * (n - ovr):j * (n - ovr) + n] += Window_occ
            itr += 1
    if TEST:
        plt.figure();
        plt.subplot(121);
        plt.imshow(R_weights, cmap="jet");
        plt.title("Weights map of the reconstructed image");
        plt.colorbar();
        plt.xlabel("x");
        plt.ylabel("y")
        plt.subplot(122);
        plt.imshow(R_occ, cmap="jet");
        plt.title("Reconstructed image's pixels occurences across the patches");
        plt.colorbar();
        plt.xlabel("x");
        plt.ylabel("y")
    return np.divide(R, R_weights)


def gaussian_window_2D(n=256, sigma=1., mu=0, m=-1, M=1, ):
    xx, yy = np.meshgrid(np.linspace(m, M, n), np.linspace(m, M, n))
    d = np.sqrt(xx * xx + yy * yy)
    return np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))

def threshold_and_clip(noisy, im, threshold=None):
    if threshold is None:
        threshold = np.mean(noisy) + 3*np.std(noisy)
    im = np.clip(im, 0, threshold)
    im = im / threshold * 255
    return im

def plot_residues(I, I_pred, res_only=False, suptitle=None):
    """ WARNING : some values are excluded with a theshold """
    Res_I = np.divide(I + 1e-10, I_pred + 1e-10)
    # Res_I = threshold(Res_I,0,np.percentile(Res_I,100-0.05)) # to exclude the 33 highest valus in the event of a big division
    # Res_I = threshold(Res_I, 0, 4)
    Res_I = np.clip(Res_I, 0, 4)
    if res_only:
        plt.figure(figsize=(12, 15))
        # plt.subplot(111, );
        plt.imshow(Res_I, cmap="gray");
        plt.title(r"Residual obtained from $I/I_{pred}$");
        plt.colorbar()
        if not (suptitle is None):
            plt.suptitle(suptitle)
        plt.tight_layout()
    else:
        plt.figure(figsize=(12, 15))
        plt.subplot(211, );
        plt.imshow(Res_I, cmap="gray");
        plt.title(r"Residual obtained from $I/I_{pred}$");
        plt.colorbar()
        plt.subplot(212);
        plt.hist(Res_I.flatten(), bins=256);
        plt.title("Residual's histogram\nmin=%.2f max=%.2f mean=%.2f std=%.2f" % (
        np.min(Res_I), np.max(Res_I), np.mean(Res_I), np.std(Res_I)));
        plt.xlabel("Values");
        plt.ylabel("Count")
        if not (suptitle is None):
            plt.suptitle(suptitle)
        plt.tight_layout()
    return Res_I



def T_EF(X, l, h, m, M):
    """ Element-wise homothety-translation for X between E=[h,l] to F=[m,M] """
    assert l < h and m <= M
    return (X - l) * (M - m) / (h - l) + m

def hamming_window(N, alpha = 0.6):
    """Generate a Hamming window."""
    n = np.arange(N)
    beta = 1-alpha
    window = alpha - beta * np.cos(2 * np.pi * n / (N-1))
    return window

def smooth(array, box, phase=False):
    """
    Imitates IDL's smooth function. Can also (correctly) smooth interferometric phases with the phase=True keyword.
    """
    if np.iscomplexobj(array):
        return filters.uniform_filter(array.real, box) + 1j * filters.uniform_filter(array.imag, box)
    elif phase is True:
        return np.angle(smooth(np.exp(1j * array), box))
    else:
        return filters.uniform_filter(array.real, box)


def list_processed_patches(file_list, patch_size=256):
    data = []
    for filename in file_list:
        print(f" Processing sublook: {filename}")
        try:
            # Load the .npy file
            loaded_data = np.load(filename)
            intensity = (np.abs(loaded_data[:, :, 0] + 1j * loaded_data[:, :, 1])) ** 2
            # Create patches from the loaded data
            patches, _, _ = create_patches_n(intensity[np.newaxis,:], patch_size)
            data.extend(patches)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    return data