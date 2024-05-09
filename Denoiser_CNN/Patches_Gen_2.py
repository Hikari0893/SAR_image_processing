import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from utils import load_sar_images
from tqdm import tqdm


main_path = os.path.dirname(os.path.abspath(__file__))[:-4]
sys.path.insert(0, main_path+"patches_folder")
sys.path.insert(0, main_path+"Data")
sys.path.insert(0, main_path+"Preprocessing__")



def test(test_files, stride, output_base_folder, pat_size=256):
    if not os.path.exists(output_base_folder):
        os.makedirs(output_base_folder)

    for idx, file in enumerate(test_files):
        # Extract a unique identifier for the image, e.g., its filename without extension
        image_id = os.path.splitext(os.path.basename(file))[0]

        # Create a folder for this image's patches
        image_output_folder = os.path.join(output_base_folder, image_id)
        if not os.path.exists(image_output_folder):
            os.makedirs(image_output_folder)

        SL = load_sar_images(file)
        if SL is None:
            print(f"Failed to load image from {file}. Skipping this file.")
            continue
        SL = SL.astype(np.float32)
        if SL.shape[-1] == 2:
            SL = np.abs(SL[:, :, :, 0] + 1j * SL[:, :, :, 1])
        SL_reshaped = SL.reshape(SL.shape[0], SL.shape[1], SL.shape[2])
        im_h, im_w = SL.shape[1:]
        x_range = range(0, max(im_h - pat_size, 1), stride)
        y_range = range(0, max(im_w - pat_size, 1), stride)

        patch_count = 0
        for x in tqdm(x_range, desc='patches'):
            for y in y_range:
                patch = SL_reshaped[:, x:x + pat_size, y:y + pat_size]
                                
                patch_file = os.path.join(image_output_folder, f'patch_{patch_count:07}.npy')
                np.save(patch_file, patch)
                patch_count += 1
        print(patch_count)
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
    plt.show()




# Example usage
dir = ['/home/tonix/Documents/Dayana/scripts/converted_image.npy']
stride = 256
output_base_folder = 'Dataset'
test(dir, stride, output_base_folder)
#patch_folder = '/home/tonix/Documents/Dayana/Dataset/sublookA_065'  # Replace with the actual path
#visualize_patches(patch_folder, num_patches=3)
print('------------------')