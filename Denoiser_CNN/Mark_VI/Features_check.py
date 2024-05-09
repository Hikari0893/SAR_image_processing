import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import math
from Mark_VI import Autoencoder_KL
from tqdm import tqdm

main_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, main_path+"/../")



# Cargar el modelo pre-entrenado
checkpoint_path = '/home/tonix/Documents/Dayana/tb_logs/relu_mse_106/version_3/checkpoints/epoch=24-step=900.ckpt'  # Asegúrate de que esta ruta sea correcta
model = Autoencoder_KL.load_from_checkpoint(checkpoint_path)



# Preparar el tensor de entrada
input_tensor = torch.from_numpy(np.load('/home/tonix/Documents/Dayana/Dataset/sublookA/patch_015341.npy')).float()
input_tensor = input_tensor.unsqueeze(0)  # Añadir una dimensión de lote
#input_tensor = input_tensor.unsqueeze(1)  # Añadir una dimensión de canal si es necesario
# Normalizar si es necesario, dependiendo de cómo fue entrenado el modelo
input_tensor = model.normalize(input_tensor)

#(1, 32, 256, 256)
skip = [input_tensor] # (1, 1, 256, 256)
output_encoder_0 = model.CNN.enc_conv0(input_tensor)
output_encoder_0 = model.CNN.enc_bn0(output_encoder_0)
output_encoder_0 = F.relu(output_encoder_0) #(32)

output_encoder_1 = model.CNN.enc_conv1(output_encoder_0)
output_encoder_1 = model.CNN.enc_bn1(output_encoder_1)
output_encoder_1 = model.CNN.enc_pool1(output_encoder_1)
output_encoder_1 = F.relu(output_encoder_1) #(64)
skip.append(output_encoder_1)

output_encoder_2 = model.CNN.enc_conv2(output_encoder_1)
output_encoder_2 = model.CNN.enc_bn2(output_encoder_2)
output_encoder_2 = model.CNN.enc_pool2(output_encoder_2)
output_encoder_2 = F.relu(output_encoder_2) #(128)
skip.append(output_encoder_2)

output_encoder_3 = model.CNN.enc_conv3(output_encoder_2)
output_encoder_3 = model.CNN.enc_bn3(output_encoder_3)
output_encoder_3 = model.CNN.enc_pool3(output_encoder_3)
output_encoder_3 = F.relu(output_encoder_3) #(128)
skip.append(output_encoder_3)

output_encoder_4 = model.CNN.enc_conv4(output_encoder_3)
output_encoder_4 = model.CNN.enc_bn4(output_encoder_4)
output_encoder_4 = model.CNN.enc_pool4(output_encoder_4)
output_encoder_4 = F.relu(output_encoder_4) #(256)
skip.append(output_encoder_4)

output_encoder_5 = model.CNN.enc_conv5(output_encoder_4)
output_encoder_5 = model.CNN.enc_bn5(output_encoder_5)
output_encoder_5 = model.CNN.enc_pool5(output_encoder_5)
output_encoder_5 = F.relu(output_encoder_5) #(256)
skip.append(output_encoder_5)

output_encoder_6 = model.CNN.enc_conv6(output_encoder_5)
output_encoder_6 = model.CNN.enc_bn6(output_encoder_6)
output_encoder_6 = model.CNN.enc_pool6(output_encoder_6)
output_encoder_6 = F.relu(output_encoder_6) #(512)
# La salida tiene forma [batch_size, channels, height, width]
# output_encoder_0.shape debería ser [1, 32, 256, 256]

output_decoder_0 = model.CNN.dec_conv0(output_encoder_6)
output_decoder_0 = model.CNN.dec_bn0(output_decoder_0)

output_decoder_1 = torch.cat([output_decoder_0, skip.pop()], dim=1)
output_decoder_1 = model.CNN.dec_conv1(output_decoder_1)
output_decoder_1 = model.CNN.dec_bn1(output_decoder_1)
output_decoder_1 = F.relu(output_decoder_1)

output_decoder_2 = torch.cat([output_decoder_1, skip.pop()], dim=1)
output_decoder_2 = model.CNN.dec_conv2(output_decoder_2)
output_decoder_2 = model.CNN.dec_bn2(output_decoder_2)
output_decoder_2 = F.relu(output_decoder_2)

output_decoder_3 = torch.cat([output_decoder_2, skip.pop()], dim=1)
output_decoder_3 = model.CNN.dec_conv3(output_decoder_3)
output_decoder_3 = model.CNN.dec_bn3(output_decoder_3)
output_decoder_3 = F.relu(output_decoder_3)

output_decoder_4 = torch.cat([output_decoder_3, skip.pop()], dim=1)
output_decoder_4 = model.CNN.dec_conv4(output_decoder_4)
output_decoder_4 = model.CNN.dec_bn4(output_decoder_4)
output_decoder_4 = F.relu(output_decoder_4)

output_decoder_5 = torch.cat([output_decoder_4, skip.pop()], dim=1)
output_decoder_5 = model.CNN.dec_conv5(output_decoder_5)
output_decoder_5 = model.CNN.dec_bn5(output_decoder_5)
output_decoder_5 = F.relu(output_decoder_5)

output_decoder_6 = torch.cat([output_decoder_5, skip.pop()], dim=1)
output_decoder_6 = model.CNN.dec_conv6(output_decoder_6)

output = model.CNN.dec_output(output_decoder_6)
# Visualiza los mapas de características del primer encoder
# Esto extraerá 32 características de la primera capa de encoder
num_features = output.shape[1]  # Esto será 32 en tu caso
title = 'S_mse_dec_7'
num_cols = 2  # Puedes ajustar esto según lo que prefieras
num_rows = math.ceil(num_features / num_cols)

fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))
    
for i in tqdm(range(num_features), desc = "Features Maps"):
    row = i // num_cols
    col = i % num_cols
    ax = axes[row, col] if num_rows > 1 else axes[col]
    feature_map = output[0, i].detach().cpu().numpy()
    ax.imshow(feature_map, cmap='gray')
    ax.axis('off')

    # Ocultar los ejes vacíos si hay menos mapas de características que subplots
    for j in range(i + 1, num_rows * num_cols):
        row = j // num_cols
        col = j % num_cols
        ax = axes[row, col] if num_rows > 1 else axes[col]
        ax.axis('off')

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(title + '.png')
    plt.show()
# Si solo hay una característica, crea solo un subplot sin usar subíndices.
"""if num_features == 1:
    fig, ax = plt.subplots(figsize=(5, 5))
    feature_map = output_encoder_3[0, 0].detach().cpu().numpy()  # Extrae el mapa de características para la característica i
    ax.imshow(feature_map, cmap='gray')  # Visualiza el mapa de características en escala de grises
    ax.axis('off')
else:
    fig, axes = plt.subplots(nrows=16, ncols=num_features//16, figsize=(num_features , num_features))  # Ajusta según el número de características
    for i in range(num_features):
        # Si hay múltiples subplots, indexa como una matriz.
        if num_features > 1:
            row = i // num_cols
            col = i % num_cols
        feature_map = output_encoder_3[0, i].detach().cpu().numpy()
    axes[row, col].imshow(feature_map, cmap='gray')
    axes[row, col].axis('off')
plt.tight_layout()
plt.savefig('mse_enc_3.png')
plt.show()

# Asumiendo que output es un tensor de forma [1, C, H, W] donde C es el número de canales
# y quieres visualizar el primer canal como un mapa de calor.
if output.dim() == 4 and output.shape[1] == 1:  # Si hay un solo canal
    output_2d = output[0, 0].detach().cpu().numpy()  # Convertir a numpy y quitar las dimensiones de batch y canal
    plt.figure()
    plt.imshow(output_2d, cmap='gray')  # Visualizar como un mapa de calor en escala de grises
    plt.colorbar()  # Opcional: Mostrar una barra de color para interpretar los valores
    plt.savefig('mse_out_cnn.png')
    plt.show()
else:
    print("Output tensor has unexpected shape:", output.shape)"""