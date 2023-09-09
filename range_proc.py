import numpy as np
import matplotlib.pyplot as plt

# Given loading code
LOWER_LIM = 4500
UPPER_LIMT = 5000
picture = np.load('converted_image.npy')
complex_image = picture[LOWER_LIM:UPPER_LIMT, LOWER_LIM:UPPER_LIMT, 0] + 1j * picture[LOWER_LIM: UPPER_LIMT, LOWER_LIM:UPPER_LIMT, 1]

# 1. Transform to frequency domain
f_sar = np.fft.fft(complex_image, axis=0)  # FFT along rows for range processing

# 2. Apply the inverse of the Hamming function
N = complex_image.shape[0]  # rows of the image for range processing
inverse_hamming = 1 / np.hamming(N)

# Apply the filter to each row
for row in range(f_sar.shape[1]):
    f_sar[:, row] *= inverse_hamming

# 3. Segment the spectrum into three sections
length = f_sar.shape[0] // 3  # Assuming divisible by 3
sections = [f_sar[:length, :], f_sar[length:2*length, :], f_sar[2*length:, :]]

# 4. Apply the Hamming function to each frequency section
for i in range(3):
    hamming_window = np.hamming(sections[i].shape[0])
    for row in range(sections[i].shape[1]):
        sections[i][:, row] *= hamming_window

# 5. Apply the inverse FFT to each section
sublooks = [np.fft.ifft(sect, axis=0) for sect in sections]  # IFFT along rows for range processing

# 6. Display each result in a subplot
plt.figure(figsize=(15,5))
for i, sublook in enumerate(sublooks):
    plt.subplot(1, 3, i+1)
    plt.imshow(np.abs(sublook), cmap='gray')  # Display magnitude of the result
    plt.title(f'Sublook {i+1}')
    plt.colorbar()
    plt.axis('off')

plt.tight_layout()
plt.show()
