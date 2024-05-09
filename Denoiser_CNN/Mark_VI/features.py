import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torchvision import transforms
from Mark_VI import Autoencoder_KL


main_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, main_path+"/../")

class FeatureExtractor(nn.Module):
    def __init__(self, model, layer_names):
        super(FeatureExtractor, self).__init__()
        self.model = model
        self.layer_names = layer_names
        self._features = {}

        for layer_name in layer_names:
            layer = dict([*self.model.named_modules()])[layer_name]
            layer.register_forward_hook(self.save_feature(layer_name))

    def save_feature(self, layer_name):
        def hook(module, input, output):
            self._features[layer_name] = output.detach()
        return hook

    def forward(self, x):
        self.model(x)
        return self._features

def visualize_feature_maps(feature_maps, num_cols=8):
    num_features = feature_maps.size(1)  # Número de mapas de características
    num_rows = num_features // num_cols + int(num_features % num_cols != 0)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 2.5))

    for i in range(num_features):
        ax = axes[i // num_cols, i % num_cols]
        feature_map = feature_maps[0, i].cpu().numpy()
        ax.imshow(feature_map, cmap='gray')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('dec_conv6.png')
    plt.show()




# Ejemplo de cómo obtener los nombres de las capas
layer_names = ['enc_conv0', 'enc_conv1', 'enc_conv2', 'dec_conv0',
               'dec_conv1', 'dec_conv2', 'dec_conv3', 'dec_conv4', 'dec_conv5', 'dec_conv6']
# Cargar el modelo pre-entrenado
checkpoint_path = '/home/tonix/Documents/Dayana/tb_logs/kullback_leibler_106/version_0/checkpoints/epoch=19-step=760.ckpt'  # Asegúrate de que esta ruta sea correcta
model = Autoencoder_KL.load_from_checkpoint(checkpoint_path)

# Preparar el tensor de entrada
input_tensor = torch.from_numpy(np.load('/home/tonix/Documents/Dayana/Dataset/sublookA/patch_015341.npy')).float()
input_tensor = input_tensor.unsqueeze(0) 
# Normalizar si es necesario, dependiendo de cómo fue entrenado el modelo
input_tensor = model.normalize(input_tensor)

# Crear el extractor de características
feature_extractor = FeatureExtractor(model.CNN, layer_names)

# Procesar una entrada a través del extractor de características
input_tensor = ... # Asume que ya tienes un tensor de entrada preparado
features = feature_extractor(input_tensor)
# Visualiza los mapas de características de la primera capa convolucional
visualize_feature_maps(features['dec_conv6'])








