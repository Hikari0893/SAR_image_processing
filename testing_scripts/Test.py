import matplotlib.pyplot as plt
import numpy as np
import json
from tqdm import tqdm
from PIL import Image
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # '1,3,4,5,6,7' for 12, '0','1','2','3' on 21

from cnn_despeckling.Jarvis import *
from cnn_despeckling.utils import normalize, denormalize

config_file = "../config.json"  # Update with your JSON file path

# Load the global parameters from the JSON file
with open(config_file, 'r') as json_file:
    global_parameters = json.load(json_file)

# Access the global parameters

archive = global_parameters['global_parameters']['ARCHIVE']
checkpoint = str(global_parameters['global_parameters']['CKPT'])
patch_folder_A = global_parameters['global_parameters']['INPUT_FOLDER']
patch_folder_B = global_parameters['global_parameters']['REFERENCE_FOLDER']
num_workers = int(global_parameters['global_parameters']['NUMWORKERS'])

# Field
patch_index = 200

# Part of city
# patch_index = 1300

# Instantiate the Model
model = Autoencoder_Wilson_Ver1.load_from_checkpoint(checkpoint)  

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

        # Update coordinates for the next patch
        current_y += stride
        if current_y + patch_size > original_width:
            current_y = 0
            current_x += stride
            if current_x + patch_size > original_height:
                break  # Finished processing all patches

    # Normalize to average overlapping regions
    reconstructed /= np.maximum(counts, 1)  # Avoid division by zero
    return np.sqrt(np.abs(reconstructed))

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
directory =str('../')
# _________________________________________________________________________________________
model.to(device)

patches2process = [400,1100]
if patch_index is not None:
    for patch_index in patches2process:
        ORIGINAL_DIMS = (256, 256)

        full_patches = "../data/patches/General_SLC/"
        org_slc_patches = [f for f in os.listdir(full_patches) if f.endswith('.npy')]
        org_slc_patches.sort(key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))  # Sorting by index
        p_index = [org_slc_patches[patch_index]]


        noisy_slc = np.squeeze(np.load(os.path.join(full_patches, p_index[0])))
        amp_full = np.sqrt(noisy_slc)
        noisy_threshold = 3 * np.mean(amp_full)  # + 3 * np.std(reconstructed_image_A)
        filename = '../results/test_full_noisy_patchnum_'+str(patch_index)
        store_data_and_plot(amp_full, noisy_threshold, filename)

        sublook_a = np.squeeze(np.load(os.path.join(patch_folder_A, p_index[0])))
        int_a = sublook_a
        sublook_a = np.sqrt(sublook_a)
        filename = '../results/test_sublookA_noisy_patchnum_'+str(patch_index)
        store_data_and_plot(sublook_a, noisy_threshold, filename)

        processed_patchesA = process_patches_with_model(patch_folder_A, model, device, desc='patches SLA',
                                                        patch_index=patch_index)
        reconstructed_image_A = reconstruct_image_from_processed_patches(processed_patchesA, ORIGINAL_DIMS,
                                                                         desc='patches SLA')

        filename = '../results/test_sublookA_filtered_patchnum_'+str(patch_index)
        store_data_and_plot(reconstructed_image_A,  noisy_threshold, filename)

        sublook_b = np.squeeze(np.load(os.path.join(full_patches, p_index[0])))
        int_b = sublook_b
        sublook_b = np.sqrt(sublook_b)
        filename = '../results/test_sublookB_noisy_patchnum_'+str(patch_index)
        store_data_and_plot(sublook_b, noisy_threshold, filename)
        processed_patchesB = process_patches_with_model(patch_folder_B, model, device, desc='patches SLB',
                                                        patch_index=patch_index)
        reconstructed_image_B = reconstruct_image_from_processed_patches(processed_patchesB, ORIGINAL_DIMS,
                                                                         desc='patches SLB')
        filename = '../results/test_sublookB_filtered_patchnum_'+str(patch_index)
        # 3 * np.mean(reconstructed_image_B)
        store_data_and_plot(reconstructed_image_B,  noisy_threshold, filename)

        sum = (reconstructed_image_A + reconstructed_image_B)/2
        filename = '../results/test_AB_patchnum'+str(patch_index)
        store_data_and_plot(sum,  noisy_threshold, filename)

else:
    ORIGINAL_DIMS = (8244,9090)
    # Plotting original and filtered amplitude images
    processed_patchesA = process_patches_with_model(patch_folder_A, model, device, desc='patches SLA', patch_index=patch_index)
    reconstructed_image_A = reconstruct_image_from_processed_patches(processed_patchesA, ORIGINAL_DIMS, desc='patches SLA')
    threshold = 3*np.mean(reconstructed_image_A)# + 3 * np.std(reconstructed_image_A)
    filename = '../test_sublookA_filtered'
    store_data_and_plot(reconstructed_image_A, threshold, filename)

    processed_patchesB = process_patches_with_model(patch_folder_B, model, device, desc='patches SLB', patch_index=patch_index)
    reconstructed_image_B = reconstruct_image_from_processed_patches(processed_patchesB, ORIGINAL_DIMS, desc='patches SLB')
    threshold = 3*np.mean(reconstructed_image_B)# + 3 * np.std(reconstructed_image_B)
    filename = '../test_sublookB_filtered'
    store_data_and_plot(reconstructed_image_B, threshold, filename)

# SLC = (reconstructed_image_A + reconstructed_image_B)/2
# threshold = np.mean(SLC) + 3 * np.std(SLC)
# filename = '/path/'
# store_data_and_plot(SLC, threshold, filename)


# ENL:
# image_sar =SLC[:,:]   # Placeholder for SAR image.
# mask = np.ones((1000, 1000), dtype=bool)  # Placeholder for mask.

# enl = calculate_enl(image_sar, mask)
# print(f"ENL: {enl}")

 
