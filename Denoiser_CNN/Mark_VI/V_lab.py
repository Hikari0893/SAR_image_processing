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
        filename = f"V_relu_{batch_size}_{epochs:2d}",
        save_top_k = -1,
        every_n_epochs = 1
    )

class SEBlock(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.LeakyReLU(inplace=True),                       #ReLU
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
class SEblock_res_Unet(nn.Module):
    # ... (tu código existente) ...
    def __init__(self, width=256, height=256):
        super(SEblock_res_Unet, self).__init__()
        
        # Tus capas existentes...
        self.enc_conv0 = nn.Conv2d(1,   32, 3, stride=1, padding=1) # (Batch, 32, 256,256)
        self.se_block0 = SEBlock(ch_in=32)
        self.enc_bn0 = nn.BatchNorm2d(32)
        
        self.enc_conv1 = nn.Conv2d(32,  64, 3, stride=1, padding=1) # (Batch, 64, 256,256)
        self.se_block1 = SEBlock(ch_in=64)
        self.enc_bn1 = nn.BatchNorm2d(64)
        self.enc_pool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0) # (Batch, 64, 128,128)
        
        self.enc_conv2 = nn.Conv2d(64,  128, 3, stride=1, padding=1) # (Batch, 128, 128,128)
        self.se_block2 = SEBlock(ch_in=128)
        self.enc_bn2 = nn.BatchNorm2d(128)
        self.enc_pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0) # (Batch, 128, 64,64)
        
        self.enc_conv3 = nn.Conv2d(128, 128, 3, stride=1, padding=1) # (Batch, 128, 64,64)
        self.se_block3 = SEBlock(ch_in=128)
        self.enc_bn3 = nn.BatchNorm2d(128)
        self.enc_pool3 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0) # Batch, 128, 32,32)
        
        self.enc_conv4 = nn.Conv2d(128, 256, 3, stride=1, padding=1) # (Batch, 256, 32,32)
        self.se_block4 = SEBlock(ch_in=256)
        self.enc_bn4 = nn.BatchNorm2d(256)
        self.enc_pool4 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0) # (Batch, 256, 16, 16)
        
        self.enc_conv5 = nn.Conv2d(256, 256, 3, stride=1, padding=1) # (Batch, 256, 16,16)
        self.se_block5 = SEBlock(ch_in=256)
        self.enc_bn5 = nn.BatchNorm2d(256)
        self.enc_pool5= nn.AvgPool2d(kernel_size=2, stride=2, padding=0) # (Batch, 256, 8, 8)
        
        self.enc_conv6 = nn.Conv2d(256, 512, 3, stride=1, padding=1) # (Batch, 512, 8,8)
        self.se_block6 = SEBlock(ch_in=512)
        self.enc_bn6 = nn.BatchNorm2d(512)
        self.enc_pool6= nn.AvgPool2d(kernel_size=2, stride=2, padding=0) # (Batch, 512, 4,4)
        
        flatten_zise = 8192 #(512*4*4)
        
        self.fc1_encoder = nn.Linear(flatten_zise, out_features=flatten_zise//2)
        self.fc2_encoder = nn.Linear(flatten_zise//2, out_features=flatten_zise//4)
        self.fc3_encoder = nn.Linear(flatten_zise//4, out_features=flatten_zise//8)
        
        # Agregar SEBlocks después de las capas convolucionales
        self.se_block1 = SEBlock(ch_in=64)
        self.se_block2 = SEBlock(ch_in=128)
        # ... agregar SE blocks para cada nivel de características ...
         # Decoder Layers
        self.fc0_decoder = nn.Linear(flatten_zise//8, out_features=flatten_zise//4) 
        self.fc1_decoder = nn.Linear(flatten_zise//4, out_features=flatten_zise//2)
        self.fc2_decoder = nn.Linear(flatten_zise//2, out_features=flatten_zise)
        
        self.dec_ups0  =  nn.Upsample(scale_factor=2, mode= 'nearest')
        self.dec_conv0 = nn.ConvTranspose2d(768, 256, 3, stride=1, padding=1) # (Batch, 256, 8, 8)
        self.dec_bn0 = nn.BatchNorm2d(256)
        
        self.dec_ups1  =  nn.Upsample(scale_factor=2, mode= 'nearest')
        self.dec_conv1 = nn.ConvTranspose2d(512, 256, 3, stride=1, padding=1) # (Batch, 512, 16, 16)
        self.dec_bn1 = nn.BatchNorm2d(256)
        
        self.dec_ups2  =  nn.Upsample(scale_factor=2, mode= 'nearest')
        self.dec_conv2 = nn.ConvTranspose2d(384, 128, 3, stride=1, padding=1) # (Batch, 128, 32, 32)
        self.dec_bn2 = nn.BatchNorm2d(128)
        
        self.dec_ups3  =  nn.Upsample(scale_factor=2, mode= 'nearest')
        self.dec_conv3 = nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1) # (Batch, 128, 64, 64)
        self.dec_bn3 = nn.BatchNorm2d(128)
        
        self.dec_ups4  = nn.Upsample(scale_factor=2, mode= 'nearest') # (Batch, 64, 128, 128)
        self.dec_conv4 = nn.ConvTranspose2d(192, 64, 3, stride=1, padding=1) # (Batch, 64, 128, 128)
        
        
        self.dec_ups5  = nn.UpsamplingBilinear2d(scale_factor=2) # (Batch, 64, 256, 256)
        
                
        self.dec_output = nn.Conv2d(96, 1, 3, padding=1) #(Batch, 1, 256, 256)
        
    def forward(self, input):
        # ... (tu código existente) ...
       
        # Aplicar SE block después de las capas convolucionales
        x = self.enc_conv0(input) # (Batch, 32, 256,256)
        x = self.se_block0(x)
        x = self.enc_bn0(x)
        skips = [x]
        
        
        x = self.enc_conv1(x)
        x = self.se_block1(x)
        x = F.leaky_relu(self.enc_bn1(x))
        x = self.enc_pool1(x) # (Batch, 64, 128,128)
        skips.append(x)
        
        
        x = self.enc_conv2(x) 
        x = self.se_block2(x)
        x = F.leaky_relu(self.enc_bn2(x))
        x = self.enc_pool2(x) # (Batch, 128, 64,64)
        skips.append(x)
       
        
        x = self.enc_conv3(x)
        x = self.se_block3(x)
        x = F.leaky_relu(self.enc_bn3(x))
        x = self.enc_pool3(x) # (Batch, 128, 32,32)
        skips.append(x)
        
        
        x = self.enc_conv4(x)
        x = self.se_block4(x)
        x = F.leaky_relu(self.enc_bn4(x))
        x = self.enc_pool4(x) # (Batch, 256, 16, 16)
        skips.append(x)
       
        
        x = self.enc_conv5(x)
        x = self.se_block5(x)
        x = F.leaky_relu(self.enc_bn5(x))
        x = self.enc_pool5(x) # (Batch, 256, 8, 8)
        skips.append(x)
       
        
        x = self.enc_conv6(x)
        x = self.se_block6(x)
        x = F.leaky_relu(self.enc_bn6(x))
        x = self.enc_pool6(x) # (Batch, 512, 4,4)
        
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1_encoder(x))
        x = F.leaky_relu(self.fc2_encoder(x))
        x = F.leaky_relu(self.fc3_encoder(x))
        
        
        #decoder
        x = F.leaky_relu(self.fc0_decoder(x))
        x = F.leaky_relu(self.fc1_decoder(x))
        x = F.leaky_relu(self.fc2_decoder(x))
        
        x = x.view(-1, 512, 4, 4)
        
        x = self.dec_ups0 (x)
        x = torch.cat([x, skips.pop()], dim=1)
        x = F.leaky_relu(self.dec_bn0(self.dec_conv0(x))) # (Batch, 256, 8, 8)
         
        
        x = self.dec_ups1 (x)
        x = torch.cat([x, skips.pop()], dim=1)
        x = F.leaky_relu(self.dec_bn1(self.dec_conv1(x) )) # (Batch, 256, 16, 16)
         
        
        x = self.dec_ups2 (x)
        x = torch.cat([x, skips.pop()], dim=1)
        x = F.leaky_relu(self.dec_bn2(self.dec_conv2(x))) # (Batch, 128, 32, 32)
        
        
        x = self.dec_ups3 (x)
        x = torch.cat([x, skips.pop()], dim=1)
        x = F.leaky_relu(self.dec_bn3(self.dec_conv3(x))) # (Batch, 64, 64, 64)
        
        
        x = self.dec_ups4 (x)
        x = torch.cat([x, skips.pop()], dim=1)
        x = F.leaky_relu(self.dec_conv4(x)) #(Batch, 32, 128, 128)
       
        
        x = self.dec_ups5 (x)
        x = torch.cat([x, skips.pop()], dim=1)
              
               
        x = self.dec_output(x)
        
        return input - x
    
class Autoencoder_V (pl.LightningModule,NPYDataLoader):
    def __init__(self, width = 256, hight = 256):
        pl.LightningModule.__init__(self)
        NPYDataLoader.__init__(self, batch_size=batch_size, num_workers=num_workers,
                               folder_A = input_folder_A, folder_B = input_folder_B, only_test=only_test)
        self.lr = learning_rate
        self.loss_f = Loss_funct()
        self.SEB_resUNET    = SEblock_res_Unet()
        # No additional layers needed here, as there is no processing, for now
        
    def forward (self, input):
        
        # Simply return the input as is
        return self.SEB_resUNET(input)
    
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
    
    
    logger = TensorBoardLogger("tb_logs", name=f"V_relu_{batch_size}_{epochs}")
    trainer = Trainer(logger=logger, fast_dev_run=False, accelerator='gpu', callbacks=[TQDMProgressBar(refresh_rate=10), checkpoint_callback], max_epochs=epochs)
    V_Net = Autoencoder_V()
    # Fit model
    trainer.fit(V_Net)    