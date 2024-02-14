import matplotlib.pyplot as plt
import numpy as np
import json
from tqdm import tqdm
from PIL import Image
import os

from Jarvis import *
config_file = "../SAR_CNN_Denoising/CONFIGURATIONS.json"  # Update with your JSON file path

# Load the global parameters from the JSON file
with open(config_file, 'r') as json_file:
    global_parameters = json.load(json_file)

# Access the global parameters

archive = global_parameters['global_parameters']['ARCHIVE']
checkpoint = str(global_parameters['global_parameters']['CKPT'])
patch_folder_A = global_parameters['global_parameters']['INPUT_FOLDER']
patch_folder_B = global_parameters['global_parameters']['REFERENCE_FOLDER']
only_test = bool(global_parameters['global_parameters']['ONLYTEST'])
num_workers = int(global_parameters['global_parameters']['NUMWORKERS'])
ratio = 0.01

# Instantiate the Model
model = Autoencoder_Wilson_Ver1.load_from_checkpoint(checkpoint)  

def process_patches_with_model(patch_folder, model, device, desc='patches SLA', ratio=ratio):
    processed_patches = []

    # Get all .npy files and sort them
    patch_files = [f for f in os.listdir(patch_folder) if f.endswith('.npy')]
    patch_files.sort(key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))  # Sorting by index

    if ratio != 1:
        num_files_to_select = int(ratio * len(patch_files))
        patch_files = patch_files[:num_files_to_select]

    for filename in tqdm(patch_files, desc='Processing patches ...'):
        # Load patch and convert to PyTorch tensor
        patch_path = os.path.join(patch_folder, filename)
        patch = np.load(patch_path)
        patch_tensor = torch.from_numpy(patch).unsqueeze(0)  # Add batch dimension

        # Move tensor to GPU if available
        patch_tensor = patch_tensor.to(device)

        # Process with the model
        with torch.no_grad():
            model.eval()
            output_tensor = model(patch_tensor)

        # Convert output tensor to NumPy array and move to CPU
        output_array = output_tensor.squeeze(0).cpu().numpy()  # Remove batch dimension
        

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
    return reconstructed  



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
directory =str('/path/')
# _________________________________________________________________________________________
model.to(device)
ORIGINAL_DIMS = (8244,9090)
processed_patchesA = process_patches_with_model(patch_folder_A, model, device, desc='patches SLA', ratio=ratio)
reconstructed_image_A = reconstruct_image_from_processed_patches(processed_patchesA, ORIGINAL_DIMS, desc='patches SLA')
threshold = np.mean(reconstructed_image_A) + 3 * np.std(reconstructed_image_A)
filename = '/path/'
store_data_and_plot(reconstructed_image_A, threshold, filename)

processed_patchesB = process_patches_with_model(patch_folder_B, model, device, desc='patches SLB', ratio=ratio)
reconstructed_image_B = reconstruct_image_from_processed_patches(processed_patchesB, ORIGINAL_DIMS, desc='patches SLB')
threshold = np.mean(reconstructed_image_B) + 3 * np.std(reconstructed_image_B)
filename = '/path/'
store_data_and_plot(reconstructed_image_B, threshold, filename)

SLC = (reconstructed_image_A + reconstructed_image_B)/2
threshold = np.mean(SLC) + 3 * np.std(SLC)
filename = '/path/'
store_data_and_plot(SLC, threshold, filename)


# ENL:
image_sar =SLC[:,:]   # Placeholder for SAR image.
mask = np.ones((1000, 1000), dtype=bool)  # Placeholder for mask.

enl = calculate_enl(image_sar, mask)
print(f"ENL: {enl}")

 
