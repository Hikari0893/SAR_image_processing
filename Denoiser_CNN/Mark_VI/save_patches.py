import matplotlib.pyplot as plt
import numpy as np
import json
from tqdm import tqdm
from PIL import Image
import matplotlib.image as mpimg
import cv2

import os
import sys

main_path = os.path.dirname(os.path.abspath(__file__))[:-7]
sys.path.insert(0, main_path+"Mark_III")


from Echo_ver5 import *
from symetriz_torch import symetrisation_patch_test

# Función para guardar las imágenes
def save_image(image, directory, filename): 
    # Asegúrate de que el directorio exista
    os.makedirs(directory, exist_ok=True)
    
    # Guarda la imagen en el directorio especificado
    filepath = os.path.join(directory, filename)
    Image.fromarray(image).save(filepath)

# Función para procesar y guardar parches
def process_and_save_patches(patch_folder, model, device, output_folder):
    patch_files = sorted([f for f in os.listdir(patch_folder) if f.endswith('.npy')],
                         key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))
    patch_count = 0
    for filename in tqdm(patch_files[0:2000], desc = 'PATCHES'):
        patch_path = os.path.join(patch_folder, filename)
        patch = np.load(patch_path).astype(np.float32)
        patch_tensor = torch.from_numpy(patch).unsqueeze(0).to(device)
        #patch_tensor = model.normalize(patch_tensor)

        with torch.no_grad():
            model.eval()
            data = model.normalize(patch_tensor)
            output_tensor = model(data)
            data       = output_tensor
            output_tensor = model.denormalize(data, model.M, model.m)
        
        # Asegúrate de que el array de salida esté en el rango correcto y sea de tipo 'uint8'
        output_array = output_tensor.cpu().numpy().squeeze()
        #filter_output = median_filter(output_array, 4)
        patch_file = os.path.join(output_folder, f'patch_{patch_count}.npy')
        np.save(patch_file, output_array)
        patch_count += 1
      
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

# Cargar la configuración global
config_path = '/home/tonix/Documents/Dayana/Denoiser_CNN/Mark_VI/MarkVI_configuration.json'
with open(config_path, 'r') as json_file:
    global_parameters = json.load(json_file)

# Acceder a los parámetros globales
archive = global_parameters['global_parameters']['ARCHIVE']
checkpoint = global_parameters['global_parameters']['CKPT']
patch_folder_A = global_parameters['global_parameters']['INPUT_FOLDER']
patch_folder_B = global_parameters['global_parameters']['REFERENCE_FOLDER']
model = Autoencoder_Echo5.load_from_checkpoint(checkpoint)

# Configurar dispositivo y modelo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Procesar y guardar parches
processed_folder = '/home/tonix/Documents/Dayana/N_Processed_patches_Denoising'
process_and_save_patches(patch_folder_A, model, device, processed_folder)

