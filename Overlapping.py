import numpy as np
import matplotlib.pyplot as plt
from patching_pre import process_blocks_slc

# Given loading code
ROW_SIZE    = 750
COLUMN_SIZE = 750
picture = np.load('tsx_hh_slc_koeln.npy')


complex_image = picture[:, :, 0] + 1j * picture[: ,: , 1]
del picture

def hamming_window(M):
    # This function returns the Hamming window of size M
    alpha = 6.00000023841857910e-1
    n = np.arange(M)
    w = alpha - (1 - alpha) * np.cos((2 * np.pi * n) / (M - 1))
    return w

R_LOWER_LIM = 24000
R_UPPER_LIMT = 27250

C_LOWER_LIM = 18000
C_UPPER_LIMT = 20000

f_sar = process_blocks_slc(complex_image, ROW_SIZE, COLUMN_SIZE, direction='row')
block_sar = f_sar[R_LOWER_LIM:R_UPPER_LIMT, C_LOWER_LIM:C_UPPER_LIMT]
del f_sar

# Inverse of Hamming window
M = block_sar.shape[1]
inverse_hamming = 1 / hamming_window(M)
for col in range(block_sar.shape[0]):
    block_sar[col] *= inverse_hamming

# Change the overlap calculation and segmentation for n sublooks
number_SL = 3  # number of sublooks, change as desired
# Segment the spectrum with overlap
overlap_factor = 0.25  # for 30% overlap
section_size = int(M / (number_SL - (number_SL - 1) * overlap_factor))
#sections = [block_sar[:, i:i+section_size] for i in range(0, M, int(section_size * (1 - overlap_factor)))]
sections = []
c = int((overlap_factor * M)/2)
a = block_sar[:,0:M//2+c]
b = block_sar[:,M//2-c:M]
sections.append(a)
sections.append(b)

del block_sar
# Apply Hamming window to each section
for section in sections:
    M_section = section.shape[1]
    hamming_win = hamming_window(M_section)
    for col in range(section.shape[0]):
        section[col] *= hamming_win

#iterate over sections
spatial_sections = [np.fft.ifft(np.fft.ifftshift(sect),  axis=1) for sect in sections]

plt.figure(figsize=(10, 5))

# Plot the first spatial section
plt.subplot(1, 2, 1)
plt.imshow(np.abs(spatial_sections[0]), cmap='gray', aspect='auto')
plt.title('Spatial Section 1')
plt.colorbar()

# Plot the second spatial section
plt.subplot(1, 2, 2)
plt.imshow(np.abs(spatial_sections[1]), cmap='gray', aspect='auto')
plt.title('Spatial Section 2')
plt.colorbar()

plt.tight_layout()
plt.show()


# Given your sections a and b in frequency domain after applying the Hamming window:
w1 = sections[0]
w2 = sections[1]

# Determine the non-overlapping and overlapping parts of w1 and w2
non_overlap_a = w1[:, :-c]
non_overlap_b = w2[:, c:]

# Determine the overlap region
overlap_a = w1[:, -c:]
overlap_b = w2[:, :c]

merged_freq = np.zeros((w1.shape[0], w1.shape[1] + w2.shape[1] - 2*c), dtype=complex)
merged_freq[:, :w1.shape[1]-c] = non_overlap_a
merged_freq[:, -w2.shape[1]+c:] = non_overlap_b

# Weighted averaging for the overlap region
alpha = np.linspace(0, 1, c)  # This creates a linear ramp from 0 to 1
merged_freq[:, w1.shape[1]-c:w1.shape[1]] = (1-alpha) * overlap_a + alpha * overlap_b

spatial_sections = np.fft.ifft(np.fft.ifftshift(merged_freq), axis=1)

# Plot the magnitude of the merged spatial section
plt.figure(figsize=(8, 6))
plt.imshow(np.abs(spatial_sections), cmap='gray', aspect='auto')
plt.title('Merged Spatial Section Magnitude')
plt.colorbar()
plt.tight_layout()
plt.show()

print("aguantame las carnitas")
"""
# Reconstruct the image
def reconstruct_from_sections(sections, c):
    
    a = sections[0][:,:-c]
    b = sections[1][:,c:]
    overlap = a.shape[1]
    h = hamming_window(overlap)
    # apply hamming window
    a *=h
    b *=h
    # column stack 
    
    
    overlap = int(sections[0].shape[1] * overlap_factor)
    no_overlap = sections[0].shape[1] - overlap

    reconstructed = sections[0][:, :no_overlap]
    for i in range(1, len(sections)):
        overlap_prev = sections[i-1][:, no_overlap:]
        overlap_current = sections[i][:, :overlap]
        weighted_overlap = overlap_prev * (1 - hamming_window(overlap)) + overlap_current * hamming_window(overlap)
        reconstructed = np.hstack((reconstructed, weighted_overlap, sections[i][:, overlap:no_overlap]))

    return reconstructed

reconstructed_image = reconstruct_from_sections(spatial_sections, overlap_factor)
del spatial_sections

# Display the reconstructed image
plt.imshow(np.abs(reconstructed_image), cmap='gray')
plt.colorbar()
plt.title('Reconstructed Image')
plt.savefig('Reconstructed Image.png')
plt.show()

"""