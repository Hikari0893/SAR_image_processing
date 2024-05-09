import matplotlib.pyplot as plt
import numpy as np
import json
from tqdm import tqdm
from PIL import Image
import os
from LSE_smoothie import smooth_filtering

from Jarvis_VI import *

# Load the global parameters from the JSON file
with open('/home/tonix/Documents/Dayana/Denoiser_CNN/Mark_VI/MarkVI_configuration.json', 'r') as json_file:
    global_parameters = json.load(json_file)

# Access the global parameters

archive = global_parameters['global_parameters']['ARCHIVE']
checkpoint = str(global_parameters['global_parameters']['CKPT'])
patch_folder_A =global_parameters['global_parameters']['INPUT_FOLDER']
patch_folder_B =global_parameters['global_parameters']['REFERENCE_FOLDER']
# Instantiate the Model
model = Autoencoder_Wilson_Gaussian.load_from_checkpoint(checkpoint)  



def process_patches_with_model(patch_folder, model, device, desc='patches SLA'):
    processed_patches = []

    # Get all .npy files and sort them
    patch_files = [f for f in os.listdir(patch_folder) if f.endswith('.npy')]
    patch_files.sort(key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))  # Sorting by index

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

# Luego, en tu código principal, llamarías a esta función para cada imagen reconstruida

      
def visualize_autoencoder_outputs(results, data_loader,num_samples=5):
    fig, axes = plt.subplots(3, num_samples, figsize=(20, 10))
    inputs, references = next(iter(data_loader))
    batch = results[0]

    for n in range(num_samples):
        img = batch[n]
        print(img.shape)
        axes[0, n].imshow(inputs[n].squeeze(), cmap='gray')
        axes[0, n].set_title('Input Sublook')
        axes[0, n].axis('off')

        axes[1, n].imshow(references[n].squeeze(), cmap='gray')
        axes[1, n].set_title('Reference Sublook')
        axes[1, n].axis('off')
        
        axes[2, n].imshow(img.squeeze(), cmap='gray')
        axes[2, n].set_title('Autoencoder Output')
        axes[2, n].axis('off')
         
    plt.tight_layout()
    plt.savefig('relu75_mse.png')
    plt.show()

def visualize_autoencoder_outputs_from_range(results, data_loader, start_idx, end_idx):
    num_samples = end_idx - start_idx
    fig, axes = plt.subplots(3, num_samples, figsize=(20, 10))
    inputs, references = next(iter(data_loader))
    batch = results[0]

    for i, n in enumerate(range(start_idx, end_idx)):
        img = batch[n].squeeze()
        axes[0, i].imshow(inputs[n].squeeze(), cmap='gray')
        axes[0, i].set_title(f'Input Sublook {n}')
        axes[0, i].axis('off')

        axes[1, i].imshow(references[n].squeeze(), cmap='gray')
        axes[1, i].set_title(f'Reference Sublook {n}')
        axes[1, i].axis('off')

        axes[2, i].imshow(img, cmap='gray')
        axes[2, i].set_title(f'Autoencoder Output {n}')
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.savefig('mse_relu75.png')
    plt.show()

def process_and_save_patches(patch_folder, model, device, output_folder, desc='.png' ):
    # Asegúrate de que el directorio de salida exista
    os.makedirs(output_folder, exist_ok=True)

    # Obtén todos los archivos .npy y ordenarlos
    patch_files = [f for f in os.listdir(patch_folder) if f.endswith('.npy')]
    patch_files.sort(key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))

    for filename in tqdm(patch_files):
        # Carga el parche y conviértelo en tensor de PyTorch
        patch_path = os.path.join(patch_folder, filename)
        patch = np.load(patch_path)
        patch_tensor = torch.from_numpy(patch).unsqueeze(0).to(device)

        # Procesa con el modelo
        with torch.no_grad():
            model.eval()
            output_tensor = model(patch_tensor)

        # Convertir el tensor de salida a array de NumPy y mover a CPU
        output_array = output_tensor.squeeze(0).cpu().numpy()
        
        # Escalar la imagen al rango [0, 255] y convertir a uint8
        output_array = np.clip(output_array, 0, 1)
        output_image = (output_array * 255).astype(np.uint8)
        im = Image.fromarray(output_image)
        
        # Guardar el parche procesado como imagen
        patch_filename = os.path.splitext(filename)[0] + '.png'  # Cambiar la extensión a .png
        im.save(os.path.join(output_folder, patch_filename))
        
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
        # Si no se proporciona una máscara, se utiliza toda la imagen.
        region = image.flatten()
    else:
        # Selecciona solo los píxeles de la región homogénea.
        region = image[mask]

    # Calcula el promedio y la varianza de la región homogénea.
    mean = np.mean(region)
    variance = np.var(region)

    # Calcula el ENL.
    enl = (mean ** 2) / variance

    return enl



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Luego, en tu código principal, llamarías a esta función para cada imagen reconstruida
directory =str('/home/tonix/Documents/Dayana')
# TONY_________________________________________________________________________________________
model.to(device)
ORIGINAL_DIMS = (8244,9090)
processed_patchesA = process_patches_with_model(patch_folder_A, model, device, desc='patches SLA')
reconstructed_image_A = reconstruct_image_from_processed_patches(processed_patchesA, ORIGINAL_DIMS, desc='patches SLA')
threshold = np.mean(reconstructed_image_A) + 3 * np.std(reconstructed_image_A)
filename = '/home/tonix/Documents/Dayana/SLC_7_SUBLOOKA_065'
store_data_and_plot(reconstructed_image_A, threshold, filename)

processed_patchesB = process_patches_with_model(patch_folder_B, model, device, desc='patches SLB')
reconstructed_image_B = reconstruct_image_from_processed_patches(processed_patchesB, ORIGINAL_DIMS, desc='patches SLB')
threshold = np.mean(reconstructed_image_B) + 3 * np.std(reconstructed_image_B)
filename = '/home/tonix/Documents/Dayana/SLC_7_SUBLOOKB_065'
store_data_and_plot(reconstructed_image_B, threshold, filename)

SLC = (reconstructed_image_A + reconstructed_image_B)/2
threshold = np.mean(SLC) + 3 * np.std(SLC)
filename = '/home/tonix/Documents/Dayana/SLC_7_SLC_065'
store_data_and_plot(SLC, threshold, filename)

# ENL:
image_sar =SLC[0:1000,0:1000]   # Placeholder for SAR image.
mask = np.ones((1000, 1000), dtype=bool)  # Placeholder for mask.

enl = calculate_enl(image_sar, mask)
print(f"ENL: {enl}")

 
