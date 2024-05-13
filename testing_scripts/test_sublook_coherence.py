import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as filters
import matplotlib.patches as patches

def gauss(array, dev, complex=False):
    gLim = 3.0
    if complex:
        return filters.gaussian_filter(array.real, dev, mode='nearest', truncate=gLim) \
            + 1j * filters.gaussian_filter(array.imag, dev, mode='nearest', truncate=gLim)
    else:
        return filters.gaussian_filter(array, dev, mode='nearest', truncate=gLim)


sla_full = np.load('../data/training/rg_sublookA_koeln.npy')
slb_full = np.load('../data/training/rg_sublookB_koeln.npy')

sla = sla_full[...,0] + 1j*sla_full[...,1]
slb = slb_full[...,0] + 1j*slb_full[...,1]

crop = [6200,7530,4450,5960]

gDev = 2
d1 = (sla[crop[0]:crop[1], crop[2]:crop[3]])
d2 = (slb[crop[0]:crop[1], crop[2]:crop[3]])
array = (gauss(d1*np.conj(d2), gDev)).real
sla_2 = (d1*np.conj(d1)).real
slb_2 = (d2*np.conj(d2)).real
coh = np.abs(array) / (np.sqrt(gauss(sla_2, gDev) * gauss(slb_2, gDev)))
coh = np.clip(np.nan_to_num(coh), 0.0, 1.0)
zoom_region = (slice(100, 200), slice(1300, 1400))

# Create a subplot with 1 row and 3 columns
plt.figure(figsize=(12, 4))  # Adjust the figure size as needed
plt.subplot(2, 3, 1)
plt.imshow(coh, cmap='gray', vmax=1)  # Display coherence data
plt.colorbar()

plt.title('Sublook Coherence')
plt.gca().add_patch(patches.Rectangle((1300, 100), 100, 100, linewidth=2, edgecolor='blue', facecolor='none'))

plt.subplot(2, 3, 2)
plt.imshow(np.abs(d1), cmap='gray', vmax=400)  # Display absolute value of complex data 1
plt.title('Sublook A Amplitude')
plt.gca().add_patch(patches.Rectangle((1300, 100), 100, 100, linewidth=2, edgecolor='blue', facecolor='none'))

plt.subplot(2, 3, 3)
plt.imshow(np.abs(d2), cmap='gray', vmax=400)  # Display absolute value of complex data 2
plt.title('Sublook B Amplitude')
plt.gca().add_patch(patches.Rectangle((1300, 100), 100, 100, linewidth=2, edgecolor='blue', facecolor='none'))

# Second Row with zoomed-in regions
plt.subplot(2, 3, 4)
plt.imshow(coh[zoom_region], cmap='gray', vmax=1)  # Display zoomed-in region of coherence data
plt.title('Coherence Zoom')
plt.colorbar()

plt.subplot(2, 3, 5)
plt.imshow(np.abs(d1)[zoom_region], cmap='gray', vmax=400)  # Display zoomed-in region of absolute value of complex data 1
plt.title('Sublook A Amplitude Zoom')

plt.subplot(2, 3, 6)
plt.imshow(np.abs(d2)[zoom_region], cmap='gray', vmax=400)  # Display zoomed-in region of absolute value of complex data 2
plt.title('Sublook B Amplitude Zoom')

plt.tight_layout()  # Adjust vertical spacing between subplots

plt.show()




