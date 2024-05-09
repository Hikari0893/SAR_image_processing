import matplotlib.pyplot as plt
import numpy as np
import json
import os
import sys

main_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, main_path+"patches_folder")
sys.path.insert(0, main_path+"Dataset")
sys.path.insert(0, main_path+"Data")
sys.path.insert(0, main_path+"Preprocessing__")
sys.path.insert(0, main_path+"Mark_V")

from Mark_V import *

# Load the global parameters from the JSON file
with open('/home/tonix/Documents/Dayana/Denoiser_CNN/Mark_V/configuration.json', 'r') as json_file:
    global_parameters = json.load(json_file)

# Access the global parameters

archive = global_parameters['global_parameters']['ARCHIVE']
checkpoint = str(global_parameters['global_parameters']['CKPT'])

# Instantiate the Model
model = Autoencoder_Tony.load_from_checkpoint(checkpoint)  


# DataLoader for test data
test_loader = model.predict_dataloader()
# Get a batch of data
images = next(iter(test_loader))
# Make sure the model is in evaluation mode and is not calculating gradients
model.eval()

with torch.no_grad():
    # Get the reconstructions
    reconstructions = model.predict_step(images,0)

    
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
    plt.savefig('loaderonly_tony.png')
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
    plt.savefig('tony_loaderonly.png')
    plt.show()

 

trainer = pl.Trainer(accelerator='gpu')
test_loader_wilson = model.predict_dataloader()
#get prediction from data loader
predictions = trainer.predict(model, dataloaders=test_loader_wilson)

# Example usage
k = 90  # Start index
q = 100  # End index (exclusive)
visualize_autoencoder_outputs_from_range(predictions, test_loader_wilson, k, q)   
visualize_autoencoder_outputs(predictions,test_loader_wilson,num_samples=8)

# Visualize the outputs of the autoencoder before training
#visualize_autoencoder_outputs(model, model.predict_dataloader(), num_samples=8)

# Visualize the outputs of the autoencoder after training
#visualize_autoencoder_outputs(model, model.train_dataloader(), num_samples=8)
#print('--------*--------*-----------')