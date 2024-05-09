import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from LSE_smoothie import smooth_filtering
import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm
from skimage.restoration import denoise_nl_means, estimate_sigma
import json

# Load global parameters from JSON file
config_file = "/home/tonix/Documents/Dayana/Denoiser_CNN/Mark_VI/MarkVI_configuration.json"  # Update with your JSON file path
with open(config_file, 'r') as json_file:
    global_parameters = json.load(json_file)

# Extract relevant parameters
M  = global_parameters['global_parameters']['M']
m  = global_parameters['global_parameters']['m']

def compare_input_output_patches(input_folder, output_folder, start_idx, end_idx):
    if start_idx >= end_idx:
        raise ValueError("El índice de inicio debe ser menor que el índice final")

    # Encuentra los archivos de parches en ambas carpetas
    input_patch_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.npy')],
                               key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))
    output_patch_files = sorted([f for f in os.listdir(output_folder) if f.endswith('.npy')],
                                key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))

    num_images = end_idx - start_idx
    num_cols = num_images  # Dos columnas: una para entrada, una para salida
    num_rows = 2  # Una fila por imagen

    plt.figure(figsize=(10, num_rows * 2))  # Ajustar tamaño de la figura

    for i, idx in enumerate(range(start_idx, end_idx)):
        # Carga parches de entrada y salida
        input_patch_path = os.path.join(input_folder, input_patch_files[idx])
        output_patch_path = os.path.join(output_folder, output_patch_files[idx])
        input_patch = np.load(input_patch_path)
        
        
        output_patch = np.load(output_patch_path)
        output_patch = np.squeeze(output_patch)
                
        #filter_patch = non_local_means_denoising(torch.from_numpy(output_patch).repeat(1,3,1,1).to(device='cuda'))
        """imagen_min = np.min(output_patch)
        imagen_max = np.max(output_patch)
        imagen_normalizada = (output_patch - imagen_min) / (imagen_max - imagen_min)
        sp = smooth_filtering(torch.from_numpy(imagen_normalizada).repeat(1,3,1,1).to(device='cuda'),
                              otf=None, lambd=0.0045, p=1.7, itr=250)"""

        # Muestra el parche de entrada
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(np.squeeze(input_patch), cmap='gray')
        plt.title(f'Input Patch {idx}')
        plt.axis('off')

        # Muestra el parche de salida
        plt.subplot(num_rows, num_cols, num_cols + i + 1)
        plt.imshow(output_patch, cmap='gray') #sp[0,0,:,:].cpu().numpy() filter_patch[0,0,:,:]
        plt.title(f'Output Patch {idx}')
        plt.axis('off')
        #TF.to_pil_image(sp[0]).show()


    plt.tight_layout()
    #plt.savefig('Co_log_ll_01.png')
    plt.show()
    
def median_filter(data, filter_size):
    temp = []
    indexer = filter_size // 2
    data_final = []
    data_final = np.zeros((len(data),len(data[0])))
    for i in range(len(data)):

        for j in range(len(data[0])):

            for z in range(filter_size):
                if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                    for c in range(filter_size):
                        temp.append(0)
                else:
                    if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                        temp.append(0)
                    else:
                        for k in range(filter_size):
                            temp.append(data[i + z - indexer][j + k - indexer])

            temp.sort()
            data_final[i][j] = temp[len(temp) // 2]
            temp = []
    return data_final

def non_local_means_denoising(img_tensor):
    """
    Aplica el filtrado no local de medias a un tensor de imagen.
    
    Args:
    img_tensor: Un tensor de PyTorch con la imagen. Debe tener dimensiones [B, C, H, W]
    
    Returns:
    Un tensor de PyTorch con la imagen filtrada.
    """
    img_np = img_tensor.cpu().numpy()  # Convertir a numpy
    denoised_imgs = []
    
    for img in img_np:
        # Estimar el sigma del ruido
        sigma_est = np.mean(estimate_sigma(img))
        
        # Aplicar el filtrado no local de medias
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
        
        imagen_min = np.min(patch)
        imagen_max = np.max(patch)
        imagen_normalizada = (patch - imagen_min) / (imagen_max - imagen_min)
        smoothie_patch = smooth_filtering(torch.from_numpy(imagen_normalizada).repeat(1,3,1,1).to(device='cuda'), otf=None, lambd=0.0045, p=1.7, itr=250)
        smoothie_patch = smoothie_patch[0,0,:,:].cpu().numpy()
        # Place the patch in the reconstructed image
        reconstructed[current_x:current_x + patch_size, current_y:current_y + patch_size] += smoothie_patch
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
input_folder = '/home/tonix/Documents/Dayana/Dataset/slc'

output_folder = '/home/tonix/Documents/Dayana/N_Processed_patches_Denoising'  
start_idx = 50
end_idx = 61

compare_input_output_patches(input_folder, output_folder, start_idx, end_idx)
"""if __name__ == '__main__':
    # Example usage:
    original_dims = (29362,17934)  # Provide the dimensions of the original image
    output_folder = '/home/tonix/Documents/Dayana/N_Processed_patches_Denoising' 
    reconstructed_image = reconstruct_image_from_processed_patches(output_folder, original_dims)
    # Save or display the reconstructed_image as needed
    
    filename = '/home/tonix/Documents/Dayana/merlin_SUBLOOK_A'  # Especifica la ruta y el nombre del archivo de salida
    threshold = np.mean(reconstructed_image) + 3 * np.std(reconstructed_image)
    im = np.clip(reconstructed_image, 0, threshold)
    im = im / threshold * 255
    im = Image.fromarray(im.astype(np.uint8)).convert('L')
    im.save(filename + '.png')"""

