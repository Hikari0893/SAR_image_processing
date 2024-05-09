import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from utils import load_sar_images


main_path = os.path.dirname(os.path.abspath(__file__))[:-4]
sys.path.insert(0, main_path+"patches_folder")
sys.path.insert(0, main_path+"Data")
sys.path.insert(0, main_path+"Preprocessing__")


def visualize_patches(folder, num_patches=5):
    """
    Visualizes a specified number of patches from a given folder.

    :param folder: Path to the folder containing patch files.
    :param num_patches: Number of patches to visualize.
    """
    patch_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.npy')]
    patch_files = patch_files[:num_patches]  # Select the first 'num_patches' files

    plt.figure(figsize=(15, 3))
    for i, patch_file in enumerate(patch_files, 1):
        patch = np.squeeze(np.load(patch_file))
        plt.subplot(1, num_patches, i)
        plt.imshow(patch, cmap='gray')
        plt.axis('off')
        plt.title(f'Patch {i}')
    #plt.savefig('patches.png')    
    plt.show()




patch_folder = '/home/tonix/Documents/Dayana/Dataset/sublookA'  # Replace with the actual path
visualize_patches(patch_folder, num_patches=8)
print('------------------')