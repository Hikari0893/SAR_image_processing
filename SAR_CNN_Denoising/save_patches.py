
import numpy as np
import json
from tqdm import tqdm
from PIL import Image

import os
import sys

main_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, main_path+"")


from Jarvis import * #import model

# Patches saving function
def save_image(image, directory, filename): 
    # Ensure that directory exists
    os.makedirs(directory, exist_ok=True)
    
    # Saving patches in the specific directory
    filepath = os.path.join(directory, filename)
    Image.fromarray(image).save(filepath)

# Process and saving
def process_and_save_patches(patch_folder, model, device, output_folder):
    patch_files = sorted([f for f in os.listdir(patch_folder) if f.endswith('.npy')],
                         key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))
    patch_count = 0
    for filename in tqdm(patch_files[0:1000], desc = 'PATCHES'):
        patch_path = os.path.join(patch_folder, filename)
        patch = np.load(patch_path)
        patch_tensor = torch.from_numpy(patch).unsqueeze(0).to(device)
        

        with torch.no_grad():
            model.eval()
            output_tensor = model(patch_tensor)
        
        # Make sure that the output array is in the correct range and is of type 'uint8'.
        output_array = output_tensor.cpu().numpy().squeeze()
        
        patch_file = os.path.join(output_folder, f'patch_{patch_count}.npy')
        np.save(patch_file, output_array)
        patch_count += 1
      

# Load global configuration
config_path = '//'
with open(config_path, 'r') as json_file:
    global_parameters = json.load(json_file)

# Access global parameters
archive = global_parameters['global_parameters']['ARCHIVE']
checkpoint = global_parameters['global_parameters']['CKPT']
patch_folder_A = global_parameters['global_parameters']['INPUT_FOLDER']
patch_folder_B = global_parameters['global_parameters']['REFERENCE_FOLDER']
model = Autoencoder_Wilson_Ver1.load_from_checkpoint(checkpoint)

# Configure device and model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Process and save patches
processed_folder = '/home/tonix/Documents/Dayana/N_Processed_patches_Denoising'
process_and_save_patches(patch_folder_A, model, device, processed_folder)
