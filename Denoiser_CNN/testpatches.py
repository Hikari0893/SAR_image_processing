
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import exposure

main_path = os.path.dirname(os.path.abspath(__file__))[:-4]
sys.path.insert(0, main_path+"patches_folder")
sys.path.insert(0, main_path+"Data")
sys.path.insert(0, main_path+"Preprocessing__")


#dir  = "/home/tonix/Documents/Dayana/Dataset/sublookB/patch_44252.npy"
dir  = "/home/tonix/Documents/Dayana/patches_folder/patch_6.npy"
patches = np.load(dir)  # Replace with your .npy file path
print(patches.shape)

# Select a patch
#patch_index = 71  # Change this to select a different patch

# patch_to_visualize has shape (1, 256, 256, 1)
patch_to_visualize = patches.squeeze() # Remove the extra dimensions
print(patch_to_visualize)

# Visualize the patch

min_val = np.min(patch_to_visualize)
max_val = np.max(patch_to_visualize)
patch_to_visualize = (patch_to_visualize - min_val) / (max_val - min_val)

threshold = np.mean(patch_to_visualize)+ 3*np.std(patch_to_visualize)
im = np.clip(patch_to_visualize,0,threshold)
im = im/threshold*255

# Normalize the image
#im = np.clip(patch_to_visualize, 0, np.max(patch_to_visualize))
#im = im / np.max(im)

# Apply histogram equalization
im_eq = exposure.equalize_hist(im,nbins=1024)
#im = Image.fromarray(im.astype('float64')).convert('L')


plt.imshow(im_eq, cmap='gray')
plt.colorbar()  # Optional: to show intensity scale
plt.title('Visualizing Patch')
plt.tight_layout()
plt.savefig('patche.png')
plt.show()

print("--------------")

