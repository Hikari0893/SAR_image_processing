import torch
import torch.nn as nn
import torch.nn.functional as F

class Activation_funct(nn.Module):
    def __init__(self):
        super(Activation_funct, self).__init__()
        #self.prelu = nn.PReLU()
    def forward(self, layer_n, function):
        
        if function == 'relu':
            return F.relu(layer_n)
        
        elif function == 'leaky_relu':
            return F.leaky_relu(layer_n)
        
        elif function == 'elu':
            return F.elu(layer_n)
        
        #elif function == 'prelu':
            #return self.prelu(layer_n)
        
        elif function == 'gelu':
            return F.gelu(layer_n)
        
        elif function == 'sigmoid':
            return F.sigmoid(layer_n)
        
        elif function == 'tanhyper':
            return F.tanh(layer_n)
        
        elif function == 'hard_tanh':
            return F.hardtanh(layer_n)
        else:
            raise ValueError(f"Unsupported activation function: {function}")