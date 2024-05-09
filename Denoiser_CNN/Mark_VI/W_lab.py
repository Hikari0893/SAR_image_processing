import os
import sys
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
import torchvision.models as models
import torch.optim as optim
from torch.optim import lr_scheduler
from Dataloader_class import NPYDataLoader
from Loss_function import *
from Activation_functions import *

main_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, main_path+"/../")

# Load global parameters from JSON file
config_file = "/home/tonix/Documents/Dayana/Denoiser_CNN/Mark_VI/config.json"  # Update with your JSON file path
with open(config_file, 'r') as json_file:
    global_parameters = json.load(json_file)

# Extract relevant parameters
L = int(global_parameters['global_parameters']['L'])
M = float(global_parameters['global_parameters']['M'])
m = float(global_parameters['global_parameters']['m'])
select = global_parameters['global_parameters']['SELECT']
function = global_parameters['global_parameters']['FUNCTION']
learning_rate = float(global_parameters['global_parameters']['learning_rate'])
batch_size = int(global_parameters['global_parameters']['batch_size'])
epochs = int(global_parameters['global_parameters']['epochs'])
only_test = bool(global_parameters['global_parameters']['ONLYTEST'])
num_workers = int(global_parameters['global_parameters']['NUMWORKERS'])
input_folder_A = global_parameters['global_parameters']['INPUT_FOLDER']
input_folder_B = global_parameters['global_parameters']['REFERENCE_FOLDER']  
ckpt_path = global_parameters['global_parameters']['CKPT'] 

checkpoint_callback = ModelCheckpoint(
        dirpath ='CKPT_checkpoints',
        filename = f"W_{batch_size}_{function}_{epochs:2d}",
        save_top_k = -1,
        every_n_epochs = 1
    )

class SEBlock(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class BasicResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leaky_relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.leaky_relu(out)

        return out
class SEBlock_Unet(nn.Module):
    def __init__(self, width=256, height=256):
        super(SEBlock_Unet, self).__init__()
        
        # Define tu arquitectura U-Net aquí ...
        # Ejemplo: capas del encoder con sus bloques SE y Residual
        self.enc_conv0 = nn.Conv2d(1, 32, 3, stride=1, padding=1)
        self.res_block0 = BasicResBlock(32, 32)
        self.se_block0 = SEBlock(32)
        self.enc_bn0 = nn.BatchNorm2d(32) # (Batch, 32, 256, 256)
        
        self.enc_conv1 = nn.Conv2d(32, 48, 3, stride=1, padding=1)
        self.res_block1 = BasicResBlock(48, 48)
        self.se_block1 = SEBlock(48)
        self.enc_bn1 = nn.BatchNorm2d(48)
        self.enc_pool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0) # (Batch, 64, 128,128)
        
        self.enc_conv2 = nn.Conv2d(48, 64, 3, stride=1, padding=1)
        self.res_block2 = BasicResBlock(64, 64)
        self.se_block2 = SEBlock(64)
        self.enc_bn2 = nn.BatchNorm2d(64)
        self.enc_pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0) # (Batch, 64, 64,64)
       
        self.enc_conv3 = nn.Conv2d(64, 80, 3, stride=1, padding=1)
        self.res_block3 = BasicResBlock(80, 80)
        self.se_block3 = SEBlock(80)
        self.enc_bn3 = nn.BatchNorm2d(80)
        self.enc_pool3 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0) # (Batch, 80, 32,32)
        
        self.enc_conv4 = nn.Conv2d(80, 96, 3, stride=1, padding=1)
        self.res_block4 = BasicResBlock(96, 96)
        self.se_block4 = SEBlock(96)
        self.enc_bn4 = nn.BatchNorm2d(96)
        self.enc_pool4 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0) # (Batch,96, 16,16)
       
        
        # (Batch,96, 16,16)
        # Definir capas del decoder
        self.dec_ups0  =  nn.Upsample(scale_factor=2, mode= 'nearest')
        self.dec_conv0 = nn.ConvTranspose2d(176, 80, 3, stride=1, padding=1) # (Batch, 80, 32, 32)
        self.dec_bn0 = nn.BatchNorm2d(80)
        
        self.dec_ups1  =  nn.Upsample(scale_factor=2, mode= 'nearest')
        self.dec_conv1 = nn.ConvTranspose2d(144, 64, 3, stride=1, padding=1) # (Batch, 64, 64, 64)
        self.dec_bn1 = nn.BatchNorm2d(64)
        
        self.dec_ups2  =  nn.Upsample(scale_factor=2, mode= 'nearest')
        self.dec_conv2 = nn.ConvTranspose2d(112, 48, 3, stride=1, padding=1) # (Batch, 48, 128, 128)
        self.dec_bn2 = nn.BatchNorm2d(48)
        
        self.dec_ups3  = nn.UpsamplingBilinear2d(scale_factor=2) # (Batch, 48, 256, 256)
        
                
        self.dec_output = nn.Conv2d(80, 1, 3, padding=1) #(Batch, 1, 256, 256)
        # ... capas del decoder como upsampling y convoluciones ...

    def forward(self, input):
        # Pasar por las capas del encoder
       
        x = self.enc_conv0(input)
        x = self.res_block0(x)  # Pasar por el bloque residual
        x = self.se_block0(x)   # Pasar por el bloque SE
        x = self.enc_bn0(x) #(32,256,256)
        skips = [x]
        
        x = self.enc_conv1(x)
        x = self.res_block1(x)  # Pasar por el bloque residual
        x = self.se_block1(x)   # Pasar por el bloque SE
        x = self.enc_bn1(x) #(48,128,128)
        x = self.enc_pool1(x)
        skips.append(x)
        
        x = self.enc_conv2(x)
        x = self.res_block2(x)  # Pasar por el bloque residual
        x = self.se_block2(x)   # Pasar por el bloque SE
        x = self.enc_bn2(x) #(64,64,64)
        x = self.enc_pool2(x)
        skips.append(x)
        
        x = self.enc_conv3(x)
        x = self.res_block3(x)  # Pasar por el bloque residual
        x = self.se_block3(x)   # Pasar por el bloque SE
        x = self.enc_bn3(x) #(80,32,32)
        x = self.enc_pool3(x)
        skips.append(x)
        
        x = self.enc_conv4(x)
        x = self.res_block4(x)  # Pasar por el bloque residual
        x = self.se_block4(x)   # Pasar por el bloque SE
        x = self.enc_bn4(x) #(96,16,16)
        x = self.enc_pool4(x)
        
        # ... resto del forward pasando por las capas del encoder, bloques SE y Residual ...
        
        # Pasar por las capas del decoder
        # ... forward pasando por las capas del decoder ...
        
        
        x = self.dec_ups0(x)
        x = torch.cat([x, skips.pop()], dim=1)
        x = self.dec_conv0(x)
        x = F.leaky_relu(self.dec_bn0(x))
        
        x = self.dec_ups1(x)
        x = torch.cat([x, skips.pop()], dim=1)
        x = self.dec_conv1(x)
        x = F.leaky_relu(self.dec_bn1(x))
        
        x = self.dec_ups2(x)
        x = torch.cat([x, skips.pop()], dim=1)
        x = self.dec_conv2(x)
        x = F.leaky_relu(self.dec_bn2(x))
        
        x = self.dec_ups3(x)
        x = torch.cat([x, skips.pop()], dim=1)
        x = self.dec_output(x)
        
        
        return input - x
    
class Autoencoder_W (pl.LightningModule,NPYDataLoader):
    def __init__(self, width = 256, hight = 256):
        pl.LightningModule.__init__(self)
        NPYDataLoader.__init__(self, batch_size=batch_size, num_workers=num_workers,
                               folder_A = input_folder_A, folder_B = input_folder_B, only_test=only_test)
        self.lr = learning_rate
        self.loss_f = Loss_funct()
        self.Hyb_Unet    = SEBlock_Unet()
        # No additional layers needed here, as there is no processing, for now
        
    def forward (self, input):
        
        # Simply return the input as is
        return self.Hyb_Unet (input)
    
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
        X_norm = self.normalize(x)
        X_true_norm = self.normalize(x_true)

        # neural network (rk is the denoised image, in the signal domain)
        X_hat = self(X_norm)

        # Denormalizing bk and rk, but still in log domain
        log_x_true = 2 * X_true_norm * (self.M - self.m) + self.m
        log_x_hat = 2 * X_hat * (self.M - self.m) + self.m

        #loss function
        loss_fuct = self.loss_f(log_x_hat, log_x_true, select = select) #The select parameter is to choose the loss function
        return loss_fuct
    
    def training_step(self,batch, batch_idx):
        loss = self.common_step(batch) 
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True) #tensorboard logs
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
        
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
    
    
    logger = TensorBoardLogger("tb_logs", name=f"W_{batch_size}_{epochs}")
    trainer = Trainer(logger=logger, fast_dev_run=False, accelerator='gpu', callbacks=[TQDMProgressBar(refresh_rate=10), checkpoint_callback], max_epochs=epochs)
    W_Net = Autoencoder_W()
    # Fit model
    trainer.fit(W_Net)      
