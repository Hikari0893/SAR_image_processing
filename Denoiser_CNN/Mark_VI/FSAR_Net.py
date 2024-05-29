import os
import sys
import torch
import torch.nn as nn
import json
from Activation_functions import *


torch.set_float32_matmul_precision('medium')

main_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, main_path + "/../")

# Load global parameters from JSON file
config_path = '/home/tonix/Documents/Dayana/Denoiser_CNN/Mark_VI/config.json'
with open(config_path, 'r') as json_file:
    global_parameters = json.load(json_file)


function = global_parameters['global_parameters']['FUNCTION']

class fsr_Unet(nn.Module):
    def __init__(self, width=96, height=96):
        super(fsr_Unet, self).__init__()
        self.activation_funct = Activation_funct() # Assume ReLU as the default activation function

        # Encoder Layers
        # [Batch_size, Channel, H, W]
        # (Batch_size, 1, 64, 64)
        self.enc_conv0 = nn.Conv2d(1, 48, 3, stride=1, padding=1)  # (Batch_size, 48, 96, 96)

        self.enc_conv1 = nn.Conv2d(48, 96, 3, stride=1, padding=1)  # (Batch_size, 96, 96, 96)
        self.enc_bn1 = nn.BatchNorm2d(96)
        self.enc_pool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)  # (Batch_size, 96, 48, 48)

        self.enc_conv2 = nn.Conv2d(96, 192, 3, stride=1, padding=1)  # (Batch_size, 192, 48, 48)
        self.enc_bn2 = nn.BatchNorm2d(192)
        self.enc_pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)  # (Batch_size, 192, 24, 24)

        self.enc_conv3 = nn.Conv2d(192, 384, 3, stride=1, padding=1)  # (Batch_size, 384, 24, 24)
        self.enc_bn3 = nn.BatchNorm2d(384)
        self.enc_pool3 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)  # (Batch_size, 384, 12, 12)

        self.enc_conv4 = nn.Conv2d(384, 768, 3, stride=1, padding=1)  # (Batch_size, 768, 12, 12)
        self.enc_bn4 = nn.BatchNorm2d(768)
        self.enc_pool4 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)  # Batch_size, 768, 6, 6)

        self.enc_conv5 = nn.Conv2d(768, 1536, 3, stride=1, padding=1)  # (Batch_size, 1536, 6, 6)
        self.enc_bn5 = nn.BatchNorm2d(1536)
        self.enc_pool5 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)  # (Batch_size, 1536, 3, 3)

        self.enc_conv6 = nn.Conv2d(1536, 2048, 4, stride=1, padding=2)  # (Batch_size, 2048, 4, 4)
        self.enc_bn6 = nn.BatchNorm2d(2048)
        self.enc_pool6 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)  # (Batch_size, 2048, 2, 2)

        flatten_size = 8192 #(2048 * 2 * 2)

        self.fc1_encoder = nn.Linear(flatten_size, out_features=flatten_size // 2)
        self.fc2_encoder = nn.Linear(flatten_size // 2, out_features=flatten_size // 4)

        # Decoder Layers
        self.fc1_decoder = nn.Linear(flatten_size // 4, out_features=flatten_size // 2)
        self.fc2_decoder = nn.Linear(flatten_size // 2, out_features=flatten_size)
        
        
        # (Batch_size, 2048, 2, 2)
        self.dec_conv0 = nn.ConvTranspose2d(2048, 1536, 3, stride=2, padding=1,
                                            output_padding=1)  # (Batch_size, 1536, 4, 4)
        self.dec_bn0 = nn.BatchNorm2d(1536)

        self.dec_conv1 = nn.ConvTranspose2d(1536, 768, 3, stride=2, padding=2,
                                            output_padding=1)  # (Batch_size, 768, 6, 6)
        self.dec_bn1 = nn.BatchNorm2d(768)

        self.dec_conv2 = nn.ConvTranspose2d(1536, 384, 3, stride=2, padding=1,
                                            output_padding=1)  # (Batch_size, 384, 12, 12)
        self.dec_bn2 = nn.BatchNorm2d(384)

        self.dec_conv3 = nn.ConvTranspose2d(768, 192, 3, stride=2, padding=1,
                                            output_padding=1)  # (Batch_size, 192, 24, 24)
        self.dec_bn3 = nn.BatchNorm2d(192)

        self.dec_conv4 = nn.ConvTranspose2d(384, 96, 3, stride=1, padding=1,
                                            output_padding=0) # (Batch_size, 96, 24, 24)

        self.dec_ups  = nn.Upsample(scale_factor=2, mode= 'nearest') # (Batch_size, 96, 48, 48)

        self.dec_conv5  = nn.UpsamplingBilinear2d(scale_factor=2) # (Batch_size, 96, 96, 96)


        self.dec_output = nn.Conv2d(240, 1, 3, padding=1)  # (Batch_size, 1, 96, 96)

    def forward(self, input):
        # Encoder
        # (Batch_size, 1, 96, 96)

        encoder_0 = self.activation_funct(self.enc_conv0(input), function)  # (Batch_size, 48, 96, 96)
        skip = [encoder_0]

        encoder_1 = self.activation_funct(self.enc_pool1(self.enc_bn1(self.enc_conv1(encoder_0))),
                                          function)   # (Batch_size, 96, 48, 48)
        skip.append(encoder_1)

        encoder_2 = self.activation_funct(self.enc_pool2(self.enc_bn2(self.enc_conv2(encoder_1))),
                                          function) # (Batch_size, 192, 24, 24)
        skip.append(encoder_2)

        encoder_3 = self.activation_funct(self.enc_pool3(self.enc_bn3(self.enc_conv3(encoder_2))),
                                          function) # (Batch_size, 384, 12, 12)
        skip.append(encoder_3)

        encoder_4 = self.activation_funct(self.enc_pool4(self.enc_bn4(self.enc_conv4(encoder_3))),
                                          function)  # Batch_size, 768, 6, 6)
        skip.append(encoder_4)

        encoder_5 = self.activation_funct(self.enc_pool5(self.enc_bn5(self.enc_conv5(encoder_4))),
                                          function)  # (Batch_size, 1536, 3, 3)
        
        encoder_6 = self.activation_funct(self.enc_pool6(self.enc_bn6(self.enc_conv6(encoder_5))),
                                          function)  # (Batch_size, 2048, 2, 2)

        encoder_6 = encoder_6.view(encoder_6.size(0), -1)
        fc1_ecoder = self.activation_funct(self.fc1_encoder(encoder_6), function)
        fc2_encoder = self.activation_funct(self.fc2_encoder(fc1_ecoder), function)

        # ... continue forward pass through additional encoder layers

        # Decoder
        # Input  (Batch_size, 512, 1, 1)
        fc1_decoder = self.activation_funct(self.fc1_decoder(fc2_encoder), function)
        fc2_decoder = self.activation_funct(self.fc2_decoder(fc1_decoder), function)
        fc_2_dec = fc2_decoder.view(-1, 2048, 2, 2)

        decoder_0 = self.activation_funct(self.dec_bn0(self.dec_conv0(fc_2_dec)), function) # (Batch_size, 1536, 4, 4)

        
        decoder_1 = self.activation_funct(self.dec_bn1(self.dec_conv1(decoder_0)), function)  # (Batch_size, 768, 6, 6)

        decoder_2 = torch.cat([decoder_1, skip.pop()], dim=1)  # Skip connection with encoder_4 # (768, 6, 6)
        decoder_2 = self.activation_funct(self.dec_bn2(self.dec_conv2(decoder_2)), function)   # (Batch_size, 384, 12, 12)

        decoder_3 = torch.cat([decoder_2, skip.pop()], dim=1)  # Skip connection with encoder_3 # (768, 12, 12)
        decoder_3 = self.activation_funct(self.dec_bn3(self.dec_conv3(decoder_3)), function)  # (Batch_size, 192, 24, 24)

        decoder_4 = torch.cat([decoder_3, skip.pop()], dim=1)  # Skip connection with encoder_2 # (192, 24, 24)
        decoder_4 = self.activation_funct(self.dec_ups(self.dec_conv4(decoder_4)), function) # (Batch_size, 96, 48, 48)

        decoder_5 = torch.cat([decoder_4, skip.pop()], dim=1)  # Skip connection with encoder_1 # (96, 48, 48)
        decoder_5 = self.activation_funct(self.dec_conv5(decoder_5), function)  # (Batch_size, 96, 96, 96)

        decoder_6 = torch.cat([decoder_5, skip.pop()], dim=1)  # Skip connection (Batch_size, 1, 96, 96)

        N = self.dec_output(decoder_6)  # (1,256,256)

        # ... continue forward pass through additional decoder layers

        return input - N