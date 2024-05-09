import os
import sys
from typing import Any
import matplotlib.pyplot as plt
#Trainer class
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
#look into progress bar
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import json


import torch.optim as optim
from torch.optim import lr_scheduler

from Dataloader_class import NPYDataLoader
from Loss_function import *
from Activation_functions import *


torch.set_float32_matmul_precision('medium')

main_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, main_path+"/../")

# Load global parameters from JSON file
config_path = '/home/tonix/Documents/Dayana/Denoiser_CNN/Mark_VI/config.json'  # Update with your JSON file path

with open(config_path, 'r') as json_file:
    global_parameters = json.load(json_file)

# Extract relevant parameters
L = int(global_parameters['global_parameters']['L'])
M = float(global_parameters['global_parameters']['M'])
m = float(global_parameters['global_parameters']['m'])
c = (1 / 2) * (special.psi(L) - np.log(L))

select = global_parameters['global_parameters']['SELECT'] #To select the activation function
function = global_parameters['global_parameters']['FUNCTION'] #Loss function
learning_rate = float(global_parameters['global_parameters']['learning_rate'])
batch_size = int(global_parameters['global_parameters']['batch_size'])
epochs = int(global_parameters['global_parameters']['epochs'])
only_test = bool(global_parameters['global_parameters']['ONLYTEST'])
num_workers = int(global_parameters['global_parameters']['NUMWORKERS'])
input_folder_A = global_parameters['global_parameters']['INPUT_FOLDER']
input_folder_B = global_parameters['global_parameters']['REFERENCE_FOLDER']  
ckpt_path = global_parameters['global_parameters']['CKPT']
ratio = global_parameters['global_parameters']['training_data_percentage']

checkpoint_callback = ModelCheckpoint(
        dirpath ='mis_checkpoints',
        filename =f"Wilson_0_relu_{epochs}_{function}",
        save_top_k = -1,
        every_n_epochs = 1
    )

class autoencoder(nn.Module):
    def __init__(self, width = 256, hight = 256):
        super(autoencoder, self).__init__()
        
        # Encoder Layers
        # [Batch_size, Channel, H, W]
        # (32, 1, 256, 256)
        self.enc_conv0 = nn.Conv2d(1,   32, 3, stride=1, padding=1) # (32, 32, 256,256)
        
        self.enc_conv1 = nn.Conv2d(32,  64, 3, stride=1, padding=1) # (32, 64, 256,256)
        self.enc_bn1 = nn.BatchNorm2d(64)
        self.enc_pool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0) 
        
        self.enc_conv2 = nn.Conv2d(64,  128, 3, stride=1, padding=1) # (32, 128, 256,256)
        self.enc_bn2 = nn.BatchNorm2d(128)
        self.enc_pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        
        self.enc_conv3 = nn.Conv2d(128, 128, 3, stride=1, padding=1) # (32, 128, 256,256)
        self.enc_bn3 = nn.BatchNorm2d(128)
        self.enc_pool3 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        
        self.enc_conv4 = nn.Conv2d(128, 256, 3, stride=1, padding=1) # (32, 256, 256,256)
        self.enc_bn4 = nn.BatchNorm2d(256)
        self.enc_pool4 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        
        self.enc_conv5 = nn.Conv2d(256, 256, 3, stride=1, padding=1) # (32, 256, 256,256)
        self.enc_bn5 = nn.BatchNorm2d(256)
        self.enc_pool5= nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        
        self.enc_conv6 = nn.Conv2d(256, 512, 3, stride=1, padding=1) # (32, 512, 256,256)
        self.enc_bn6 = nn.BatchNorm2d(512)
        self.enc_pool6= nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        
        # Decoder Layers
        self.dec_conv0 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding = 1) # (32, 256, 4, 4)
        self.dec_bn0   =  nn.BatchNorm2d(256)
        
        self.dec_conv1 = nn.ConvTranspose2d(512, 256, 2, stride=2, padding=0, output_padding=0) # (32, 512, 8, 8)
        self.dec_bn1   =  nn.BatchNorm2d(256)
        
        self.dec_conv2 = nn.ConvTranspose2d(512, 128, 2, stride=2, padding=0, output_padding=0) # (32, 128, 16, 16)
        self.dec_bn2   =  nn.BatchNorm2d(128)
        
        self.dec_conv3 = nn.ConvTranspose2d(256, 128, 2, stride=2, padding=0, output_padding=0) # (32, 128, 32, 32)
        self.dec_bn3   =  nn.BatchNorm2d(128)
        
        self.dec_conv4 = nn.ConvTranspose2d(256, 64, 2, stride=2, padding=0, output_padding=0) # (32, 64, 64, 64)
        self.dec_bn4   =  nn.BatchNorm2d(64)
        
        self.dec_conv5 = nn.ConvTranspose2d(128, 32, 2, stride=2, padding=0, output_padding=0) # (32, 32, 128, 128)
        self.dec_bn5   =  nn.BatchNorm2d(32)
        
        self.dec_conv6 = nn.ConvTranspose2d(64, 1, 1, stride=1, padding=0, output_padding = 0) # (32, 1, 256, 256)
        
        self.dec_output = nn.Conv2d(33, 1, 3, padding=1)
        
        
        
    def forward (self, input):
        # Encoder
        skip = [input] # (32, 1, 256, 256)
        encoder_0 = F.leaky_relu(self.enc_conv0(input)) # (32, 32, 256, 256)
        encoder_1 = self.enc_pool1(F.leaky_relu(self.enc_bn1(self.enc_conv1(encoder_0)))) # (32, 64, 128, 128)
        skip.append(encoder_1)
        encoder_2 = self.enc_pool2(F.leaky_relu(self.enc_bn2(self.enc_conv2(encoder_1)))) # (32, 128, 64, 64)
        skip.append(encoder_2)
        encoder_3 = self.enc_pool3(F.leaky_relu(self.enc_bn3(self.enc_conv3(encoder_2)))) # (32, 128, 32, 32)
        skip.append(encoder_3)
        encoder_4 = self.enc_pool4(F.leaky_relu(self.enc_bn4(self.enc_conv4(encoder_3)))) # (32, 256, 16, 16)
        skip.append(encoder_4)
        encoder_5 = self.enc_pool5(F.leaky_relu(self.enc_bn5(self.enc_conv5(encoder_4)))) # (32, 256, 8, 8)
        skip.append(encoder_5)
        encoder_6 = self.enc_pool6(F.leaky_relu(self.enc_bn6(self.enc_conv6(encoder_5)))) # (32, 512, 4, 4)
        #skip.append(encoder_6)

        # ... continue forward pass through additional encoder layers

        # Decoder 
        # Input  (32, 512, 4, 4)
        
        decoder_0 = F.leaky_relu(self.dec_bn0(self.dec_conv0(encoder_6)))
        
        decoder_1 = torch.cat([decoder_0, skip.pop()], dim=1)  # Skip connection
        decoder_1 = F.leaky_relu(self.dec_bn1(self.dec_conv1(decoder_1)))
        
        decoder_2 = torch.cat([decoder_1, skip.pop()], dim=1)  # Skip connection
        decoder_2 = F.leaky_relu(self.dec_bn2(self.dec_conv2(decoder_2)))
        
        decoder_3 = torch.cat([decoder_2, skip.pop()], dim=1)  # Skip connection
        decoder_3 = F.leaky_relu(self.dec_bn3(self.dec_conv3(decoder_3)))
        
        decoder_4 = torch.cat([decoder_3, skip.pop()], dim=1)  # Skip connection
        decoder_4 = F.leaky_relu(self.dec_bn4(self.dec_conv4(decoder_4)))
        
        decoder_5 = torch.cat([decoder_4, skip.pop()], dim=1)  # Skip connection
        decoder_5 = F.leaky_relu(self.dec_bn5(self.dec_conv5(decoder_5)))
        
        decoder_6 = torch.cat([decoder_5, skip.pop()], dim=1)  # Skip connection
        
        N = self.dec_output(decoder_6)

        # ... continue forward pass through additional decoder layers
                
        return input - N
        #return N
    
    
class Autoencoder_Wilson (pl.LightningModule,NPYDataLoader):
    def __init__(self, width = 256, hight = 256):
        pl.LightningModule.__init__(self)
        NPYDataLoader.__init__(self, batch_size=batch_size, num_workers=num_workers, 
                               folder_A = input_folder_A, folder_B = input_folder_B, only_test=only_test)
        self.lr = learning_rate
        self.loss_f = Loss_funct()
        self.W_autoencoder = autoencoder()
        # No additional layers needed here, as there is no processing, for now
        
    def forward (self, input):
        
        # Simply return the input as is
        return self.W_autoencoder(input)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = {
            'scheduler': lr_scheduler.MultiStepLR(optimizer, milestones=[5, 20], gamma=0.1),
            'interval': 'epoch',  # Adjust the learning rate at the end of each epoch
        }
        return [optimizer], [scheduler]
    
    def normalize(self, batch):
        #remove this after dataset
        self.M  = M
        self.m  = m
        return (torch.log(batch + 1e-7) - 2*self.m)/(2*(self.M - self.m))

    # Not used given the architecture...
    def denormalize(self,normalized_data, Max, min):
        log_data = normalized_data * (Max - min) + min
        original_data = torch.exp(log_data) - 1e-7
        return original_data

    def common_step(self,batch):
        # This defines how to handle a training step
        # If we define the input pair as (ak,bk) and output as rk
        # These are the sublook(a,b) intensities
        x, x_true = batch

        # We normalize sublook_a
        ak = self.normalize(x)
        bk = self.normalize(x_true)

        # import matplotlib.pyplot as plt
        # import numpy as np
        # Ynpy = x[0,0,:,:].cpu().numpy()
        # Xnpy = x_true[0,0,:,:].cpu().numpy()
        #
        # plt.imshow((Ynpy)**(1/2), cmap="gray", vmax=300)
        # plt.figure()
        # plt.imshow((Xnpy)**(1/2), cmap="gray", vmax=300)
        # plt.show()
        #
        # a=2

        # neural network (rk is the denoised image, in the signal domain)
        rk = self(ak)

        # Denormalizing bk and rk, but still in log domain
        log_bk = 2 * bk * (self.M - self.m) + self.m
        log_rk = 2 * rk * (self.M - self.m) + self.m

        #loss function
        loss_fuct = self.loss_f(log_rk, log_bk, select = select) #The select parameter is to choose the loss function
        return loss_fuct
    
    def training_step(self,batch, batch_idx):
        loss = self.common_step(batch) 
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True) #tensorboard logs
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    #insert tensorboard metadata
    def on_validation_epoch_end(self):
        if hasattr(self, 'validation_outputs'):
            # Concatenar los outputs
            all_outputs = torch.cat(self.validation_outputs, dim=0)

            # Calcular métricas, por ejemplo, la media
            avg_validation_loss = all_outputs.mean()

            # Registrar la métrica
            self.log('val_loss', avg_validation_loss)

            # Limpiar la lista para la próxima época
            self.validation_outputs.clear()
        else:
            # Manejar el caso sin outputs
            self.log('val_loss', 0)
    
    def predict_step(self, batch, batch_idx):
        # Assuming batch comes as a tuple (input_sublook, reference_sublook)
        input_sublook, _ = batch  # We only need the input for prediction, not the reference
        
        input_sublook = self.normalize(input_sublook)
        reconstructions = self(input_sublook)
        denormalized_reconstructions = self.denormalize(reconstructions, self.M, self.m)
        original_shape_reconstructions = denormalized_reconstructions.view(-1, 256, 256, 1)
        
        return original_shape_reconstructions

        
        
    def train_dataloader(self):
        return self.train_loader
        
    def val_dataloader(self):
        return self.val_loader
    
    def predict_dataloader(self):
        return self.test_loader
    
    
if __name__ == '__main__':
    
    
    logger = TensorBoardLogger("tb_logs", name=f"Wilson_0_relu_{epochs}_{function}")
    trainer = Trainer(logger=logger, fast_dev_run=False, accelerator='gpu', callbacks=[TQDMProgressBar(refresh_rate=10), checkpoint_callback], max_epochs=epochs)
    Wilson_0 = Autoencoder_Wilson()
    trainer.fit(Wilson_0)
    
    