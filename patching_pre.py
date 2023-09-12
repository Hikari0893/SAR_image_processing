import numpy as np
from scipy.fft import fft, fftshift
import matplotlib.pyplot as plt

"""# Simulate a SLC SAR image
m, n = 29868, 21700
sar_image = np.random.rand(m, n) + 1j * np.random.rand(m, n)  # SLC SAR image is complex-valued"""

def process_blocks_slc(image, block_size_q, block_size_p, direction='column'):
    """
    Processes a SLC SAR image block-wise applying FFT on specified direction.
    Parameters:
    - image: Input SLC SAR image of size mxn.
    - block_size_q: Block size along rows.
    - block_size_p: Block size along columns.
    - direction: Either 'column' or 'row' indicating the direction of FFT.
    Returns the FFT processed image of the same size.
    """
    # Determine the required padding
    pad_m = (block_size_q - (image.shape[0] % block_size_q)) % block_size_q
    pad_n = (block_size_p - (image.shape[1] % block_size_p)) % block_size_p
    
    # Pad the image
    padded_image = np.pad(image, ((0, pad_m), (0, pad_n)), mode='constant')

    # Pre-allocate result array
    processed_image = np.empty_like(padded_image, dtype=complex)
    
    num_blocks_m = padded_image.shape[0] // block_size_q
    num_blocks_n = padded_image.shape[1] // block_size_p

    for i in range(num_blocks_m):
        for j in range(num_blocks_n):
            start_row_idx = i * block_size_q
            end_row_idx = start_row_idx + block_size_q
            start_col_idx = j * block_size_p
            end_col_idx = start_col_idx + block_size_p

            # Extract block
            block = padded_image[start_row_idx:end_row_idx, start_col_idx:end_col_idx]

            # Process the block
            axes_dict = {'column':0,'row':1}
            block_fft = fft(block, axis=axes_dict[direction])
            block_fftshift = fftshift(block_fft, axes=axes_dict[direction])

            # Store the processed block in the result array
            processed_image[start_row_idx:end_row_idx, start_col_idx:end_col_idx] = block_fftshift

    return processed_image

