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
#from pytorch_lightning.tuner import Tuner

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
        filename =f"Z_Net_{select}_{function}_{batch_size}_{epochs}",
        save_top_k = -1,
        every_n_epochs = 1
    )


class U_net(nn.Module):
    def __init__(self, width = 256, hight = 256):
        super(U_net, self).__init__()
        self.activation_funct = Activation_funct()
        self.dropout = nn.Dropout(p=0.1)
        
        # Encoder Layers
        # [Batch_size, Channel, H, W]
        # (32, 1, 256, 256)
        self.enc_conv0 = nn.Conv2d(1,   32, 3, stride=1, padding=1) # (32, 32, 256,256)
        
        self.enc_conv1 = nn.Conv2d(32,  64, 3, stride=1, padding=1) # (32, 64, 256,256)
        self.enc_bn1 = nn.BatchNorm2d(64)
        self.enc_pool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0) # (32, 64, 128,128)
        
        self.enc_conv2 = nn.Conv2d(64,  128, 3, stride=1, padding=1) # (32, 128, 128,128)
        self.enc_bn2 = nn.BatchNorm2d(128)
        self.enc_pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0) # (32, 128, 64,64)
        
        self.enc_conv3 = nn.Conv2d(128, 128, 3, stride=1, padding=1) # (32, 128, 64,64)
        self.enc_bn3 = nn.BatchNorm2d(128)
        self.enc_pool3 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0) # (32, 128, 32,32)
        
        self.enc_conv4 = nn.Conv2d(128, 256, 3, stride=1, padding=1) # (32, 256, 32,32)
        self.enc_bn4 = nn.BatchNorm2d(256)
        self.enc_pool4 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0) # (32, 256, 16, 16)
        
        self.enc_conv5 = nn.Conv2d(256, 256, 3, stride=1, padding=1) # (32, 256, 16,16)
        self.enc_bn5 = nn.BatchNorm2d(256)
        self.enc_pool5= nn.AvgPool2d(kernel_size=2, stride=2, padding=0) # (32, 256, 8, 8)
        
        self.enc_conv6 = nn.Conv2d(256, 512, 3, stride=1, padding=1) # (32, 512, 8,8)
        self.enc_bn6 = nn.BatchNorm2d(512)
        self.enc_pool6= nn.AvgPool2d(kernel_size=2, stride=2, padding=0) # (32, 512, 4,4)
        
        flatten_zise = 8192 #(512*4*4)
        
        self.fc1_encoder = nn.Linear(flatten_zise, out_features=flatten_zise//2)
        self.fc2_encoder = nn.Linear(flatten_zise//2, out_features=flatten_zise//4)
        
        # Decoder Layers
        self.fc1_decoder = nn.Linear(flatten_zise//4, out_features=flatten_zise//2)
        self.fc2_decoder = nn.Linear(flatten_zise//2, out_features=flatten_zise)
        
        self.dec_ups0   =  nn.Upsample(scale_factor=2, mode= 'nearest')
        self.dec_conv0 = nn.Conv2d(512, 256, 3, stride=1, padding=1) # (32, 256, 8, 8)
        self.dec_bn0 = nn.BatchNorm2d(256)
        
        self.dec_ups1   =  nn.Upsample(scale_factor=2, mode= 'nearest')
        self.dec_conv1 = nn.Conv2d(512, 256, 3, stride=1, padding=1) # (32, 512, 16, 16)
        self.dec_bn1 = nn.BatchNorm2d(256)
        
        self.dec_ups2   =  nn.Upsample(scale_factor=2, mode= 'nearest')
        self.dec_conv2 = nn.Conv2d(512, 128, 3, stride=1, padding=1) # (32, 128, 32, 32)
        self.dec_bn2 = nn.BatchNorm2d(128)
        
        self.dec_ups3   =  nn.Upsample(scale_factor=2, mode= 'nearest')
        self.dec_conv3 = nn.Conv2d(256, 128, 3, stride=1, padding=1) # (32, 128, 64, 64)
        self.dec_bn3 = nn.BatchNorm2d(128)
        
        self.dec_ups4  = nn.Upsample(scale_factor=2, mode= 'nearest') # (32, 64, 128, 128)
        self.dec_conv4 = nn.Conv2d(256, 64, 3, stride=1, padding=1) # (32, 64, 64, 64)
        self.dec_bn4 = nn.BatchNorm2d(64)
        
        self.dec_ups5  = nn.UpsamplingBilinear2d(scale_factor=2) # (32, 128, 256, 256)
        
        #self.dec_conv6 = nn.ConvTranspose2d(64, 1, 1, stride=1, padding=0, output_padding = 0) # (32, 1, 256, 256)
        
        #(32,160,256,256)
        self.dec_output = nn.Conv2d(160, 1, 3, padding=1) 
        
        
        
    def forward (self, input):
        # Encoder
         # (32, 1, 256, 256)
        encoder_0 =self.activation_funct(self.enc_conv0(input), function) # (32, 32, 256, 256)
        skip = [encoder_0]
        
        encoder_1 = self.activation_funct(self.enc_pool1(self.enc_bn1(self.enc_conv1(encoder_0))), function) # (32, 64, 128, 128)
        skip.append(encoder_1)
        
        encoder_2 = self.activation_funct(self.enc_pool2(self.enc_bn2(self.enc_conv2(encoder_1))), function) # (32, 128, 64, 64)
        skip.append(encoder_2)
        
        encoder_3 = self.activation_funct(self.enc_pool3(self.enc_bn3(self.enc_conv3(encoder_2))), function) # (32, 128, 32, 32)
        skip.append(encoder_3)
        
        encoder_4 = self.activation_funct(self.enc_pool4(self.enc_bn4(self.enc_conv4(encoder_3))), function) # (32, 256, 16, 16)
        skip.append(encoder_4)
        
        encoder_5 = self.activation_funct(self.enc_pool5(self.enc_bn5(self.enc_conv5(encoder_4))), function) # (32, 256, 8, 8)
        skip.append(encoder_5)
        
        encoder_6 = self.activation_funct(self.enc_pool6(self.enc_bn6(self.enc_conv6(encoder_5))), function) # (32, 512, 4, 4)
        
        encoder_6 = encoder_6.view(encoder_6.size(0), -1)
        fc1_ecoder = self.activation_funct(self.fc1_encoder(encoder_6), function)
        fc2_encoder = self.activation_funct(self.fc2_encoder(fc1_ecoder), function)
        #skip.append(encoder_6)

        # ... continue forward pass through additional encoder layers

        # Decoder 
        # Input  (32, 512, 4, 4)
        fc1_decoder = self.activation_funct(self.fc1_decoder(fc2_encoder), function)
        fc2_decoder = self.activation_funct(self.fc2_decoder(fc1_decoder), function)
        fc_2_dec = fc2_decoder.view(-1, 512, 4, 4)
        
        decoder_0 = self.activation_funct(self.dec_bn0(self.dec_conv0(self.dec_ups0(fc_2_dec))), function)
        
        decoder_1 = torch.cat([decoder_0, skip.pop()], dim=1)  # Skip connection (512,8,8)
        decoder_1 = self.activation_funct(self.dec_bn1(self.dec_conv1(self.dec_ups1(decoder_1))), function) #(256,16,16)
        
        decoder_2 = torch.cat([decoder_1, skip.pop()], dim=1)  # Skip connection (512,16,16)
        decoder_2 = self.activation_funct(self.dec_bn2(self.dec_conv2(self.dec_ups2(decoder_2))), function) #(128,32,32)
        
        decoder_3 = torch.cat([decoder_2, skip.pop()], dim=1)  # Skip connection (256,32,32)
        decoder_3 = self.activation_funct(self.dec_bn3(self.dec_conv3(self.dec_ups3(decoder_3))), function) #(128,64,64)
        
        decoder_4 = torch.cat([decoder_3, skip.pop()], dim=1)  # Skip connection (256,64,64)
        decoder_4 = self.activation_funct(self.dec_bn4(self.dec_conv4(self.dec_ups4(decoder_4))), function) #(64,128,128)
        
        decoder_5 = torch.cat([decoder_4, skip.pop()], dim=1)  # Skip connection (128,128,128)
        decoder_5 = self.activation_funct(self.dec_ups5(decoder_5), function) #(128,256,256)
        
        decoder_6 = torch.cat([decoder_5, skip.pop()], dim=1)  # Skip connection (160, 256,256)
        
        N = self.dec_output(decoder_6) #(1,256,256)

        # ... continue forward pass through additional decoder layers
                
        return input - N
        #return N
    
    
class Autoencoder_Z (pl.LightningModule,NPYDataLoader):
    def __init__(self, width = 256, hight = 256):
        pl.LightningModule.__init__(self)
        NPYDataLoader.__init__(self, batch_size=batch_size, num_workers=num_workers, 
                               folder_A = input_folder_A, folder_B = input_folder_B, only_test=only_test, ratio=ratio)
        self.lr = learning_rate
        self.loss_f = Loss_funct()
        
        self.dnn    = U_net()
        # No additional layers needed here, as there is no processing, for now
        
    def forward (self, input):
        
        # Simply return the input as is
        return self.dnn(input)
    
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
        
        inputs = self.normalize(batch)
        reconstructions = self(inputs)
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
    
    logger = TensorBoardLogger("tb_logs", name=f"Z_Net_{select}_{function}_{batch_size}_{epochs}")
    
    trainer = Trainer(logger=logger, fast_dev_run=False, accelerator='gpu',
                      callbacks=[TQDMProgressBar(refresh_rate=10), checkpoint_callback], 
                      max_epochs=epochs)
    Z_Net = Autoencoder_Z()
       
    trainer.fit(Z_Net)#, ckpt_path=ckpt_path
    
    