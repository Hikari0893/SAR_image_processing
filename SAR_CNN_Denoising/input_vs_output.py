import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import torch
from tqdm import tqdm
from skimage.restoration import denoise_nl_means, estimate_sigma

def compare_input_output_patches(input_folder, output_folder, start_idx, end_idx):
    if start_idx >= end_idx:
        raise ValueError("El índice de inicio debe ser menor que el índice final")

    # Finds the patch files in both folders
    input_patch_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.npy')],
                               key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))
    output_patch_files = sorted([f for f in os.listdir(output_folder) if f.endswith('.npy')],
                                key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))

    num_images = end_idx - start_idx
    num_cols = num_images  # One column per image
    num_rows = 2  # Two rows: one for input, one for output

    plt.figure(figsize=(10, num_rows * 2))  # Adjust figure size Ajustar tamaño de la figura

    for i, idx in enumerate(range(start_idx, end_idx)):
        # Load input and output patches
        input_patch_path = os.path.join(input_folder, input_patch_files[idx])
        output_patch_path = os.path.join(output_folder, output_patch_files[idx])
        input_patch = np.load(input_patch_path)
        output_patch = np.load(output_patch_path)
        output_patch = np.squeeze(output_patch)
        
        # Displays input patches
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(np.squeeze(input_patch), cmap='gray')
        plt.title(f'Input Patch {idx}')
        plt.axis('off')

        # Displays output patches
        plt.subplot(num_rows, num_cols, num_cols + i + 1)
        plt.imshow(output_patch, cmap='gray') 
        plt.title(f'Output Patch {idx}')
        plt.axis('off')
      
    plt.tight_layout()
    plt.show()
    


def non_local_means_denoising(img_tensor):
    """
    Applies non-local filtering of averages to an image tensor.
    
    Args:
    img_tensor: A PyTorch tensor with the image. Must have dimensions [B, C, H, W].
    
    Returns:
    A PyTorch tensor with the filtered image.
    """
    img_np = img_tensor.cpu().numpy()  # Convert to numpy
    denoised_imgs = []
    
    for img in img_np:
        # Estimate noise sigma
        sigma_est = np.mean(estimate_sigma(img))
        
        # Apply non-local filtering of averages
        img_denoised = denoise_nl_means(img, h=1.15 * sigma_est, fast_mode=True, patch_size=5, patch_distance=6)
        denoised_imgs.append(img_denoised)
    
    return torch.from_numpy(np.array(denoised_imgs)).float()

def reconstruct_image_from_processed_patches(output_folder, original_dims, patch_size=256, stride=64, desc='patches SLA'):
    (original_height, original_width) = original_dims
    reconstructed = np.zeros((original_height, original_width), dtype=np.float32)
    counts = np.zeros_like(reconstructed, dtype=np.int32)

    current_x, current_y = 0, 0
    processed_patches = sorted([f for f in os.listdir(output_folder) if f.endswith('.npy')],
                                key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))

    for patch_filename in tqdm(processed_patches, desc=desc):
        # Load and process the patch
        output_patch_path = os.path.join(output_folder, patch_filename)
        patch = np.load(output_patch_path)
        
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



input_folder = '//'
output_folder = '//'  
start_idx =  651
end_idx = 660

compare_input_output_patches(input_folder, output_folder, start_idx, end_idx)
