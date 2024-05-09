import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys
from tqdm import tqdm

main_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, main_path+"/../Preprocessing__")



def reconstruct_image_from_patches(patch_folder, original_dims, patch_size=256, stride=64):
    patch_files = [os.path.join(patch_folder, f) for f in os.listdir(patch_folder) if f.endswith('.npy')]
    patch_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))  # Sort by patch index

    (original_height, original_width) = original_dims
    reconstructed = np.zeros((original_height, original_width), dtype=np.float32)
    counts = np.zeros_like(reconstructed, dtype=np.int32)

    current_x, current_y = 0, 0
    for n in tqdm(patch_files):
        patch = np.load(n).squeeze()
        
        # Place the patch in the reconstructed image
        reconstructed[current_x:current_x + patch_size, current_y:current_y + patch_size] += patch
        counts[current_x:current_x + patch_size, current_y:current_y + patch_size] += 1

        # Update coordinates for the next patch
        current_y += stride
        if current_y + patch_size > original_width:
            current_y = 0
            current_x += stride
            if current_x + patch_size > original_height:
                break  # Finished processing all patches

    # Normalize to average overlapping regions
    reconstructed /= np.maximum(counts, 1)  # Avoid division by zero
    return reconstructed

def list_n_patch_names(patch_folder, n):
    # Get all filenames in the patch folder that end with '.npy'
    patch_files = [f for f in os.listdir(patch_folder) if f.endswith('.npy')]
    
    # Sort the filenames
    patch_files.sort(key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))
    

    # Select the first n filenames
    return patch_files[:n]

def store_data_and_plot(im, threshold, filename):
    im = np.clip(im, 0, threshold)
    im = im / threshold * 255
    im = Image.fromarray(im.astype(np.uint8)).convert('L')
    im.save(filename + '.png')

# Example usage
original_dims = (29828, 21036)  # Height and width of the original image
patch_folder = '/home/tonix/Documents/Dayana/Dataset/sublookA'
reconstructed_image = reconstruct_image_from_patches(patch_folder, original_dims)

threshold = np.mean(reconstructed_image) + 3 * np.std(reconstructed_image)
filename = '/home/tonix/Documents/Dayana/Look_A'
store_data_and_plot(reconstructed_image, threshold, filename)

# Visualize a portion of the reconstructed image
plt.imshow(reconstructed_image[9500:10000, 12500:13000], cmap='gray')
plt.axis('off')
plt.savefig('img_zoomed.png')
plt.show()

# List patch names example
n = 10  # Number of patch names you want to see
patch_names = list_n_patch_names(patch_folder, n)
print("List of patch names:")
for name in patch_names:
    print(name)
