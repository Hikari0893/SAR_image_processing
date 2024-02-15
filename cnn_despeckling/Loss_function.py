import torch
import torch.nn as nn
import torch.nn.functional as F
import json
# Load global parameters from JSON file
config_file = "../config.json"  # Update with your JSON file path
with open(config_file, 'r') as json_file:
    global_parameters = json.load(json_file)

# Extract relevant parameters
Max = global_parameters['global_parameters']['M']
min = global_parameters['global_parameters']['m']

class Loss_funct(nn.Module):
    def __init__(self):
        super(Loss_funct, self).__init__()

    def forward(self, X_hat, Y_reference, select):
        """
        Apply the selected loss function to the input data.
        
        Args:
        X_hat (Tensor): The predicted values (output from the network).
        Y_reference (Tensor): The ground truth values.
        select (str): A string to select the loss function.
        
        Returns:
        Tensor: The computed loss.
        """
        if select == 'mse':
            return F.mse_loss(X_hat, Y_reference)
        
        elif select == 'mae':
            return F.l1_loss(X_hat, Y_reference)
        
        elif select == 'cross_entropy':
            return F.cross_entropy(X_hat, Y_reference)
        
        elif select == 'neg_log_likelihood':
            return F.nll_loss(X_hat, Y_reference)
        
        elif select == 'co_log_likelihood':
            return self.co_log_likelihood_loss(X_hat, Y_reference)

        elif select == 'co_log_likelihood_v2':
            return self.co_log_likelihood_loss_v2(X_hat, Y_reference)

        elif select == 'kullback_leibler':
            return F.kl_div(X_hat, Y_reference, reduction='batchmean')
        
        elif select == 'custom_cross_entropy':
            return self.custom_cross_entropy(X_hat, Y_reference)
        
        elif select == 'jensen_shannon_divergence':
            return self. JD_div(X_hat, Y_reference)
        
        elif select == 'L1-L2':
            return 0.6*F.mse_loss(X_hat, Y_reference) + 0.4*F.l1_loss(X_hat, Y_reference)
        
        elif select == 'HuberLoss':
            return F.huber_loss(X_hat, Y_reference)
        
        else:
            raise ValueError(f"Unsupported loss function: {select}")
    
    
    def custom_cross_entropy(self, X_hat, Y_reference):
        # Compute the negative log-likelihood
        neg_log_likelihood = -(Y_reference - X_hat + torch.exp(Y_reference - X_hat) + torch.exp(torch.exp(Y_reference - X_hat)))
        
        # Sum up the negative log-likelihood for all data points (assuming they are independent)
        total_neg_log_likelihood = torch.mean(neg_log_likelihood)
        
        # Return the negative log-likelihood as the loss
        loss = total_neg_log_likelihood
        
        return loss

    def co_log_likelihood_loss(self, X_hat, Y_reference):
        
        # Assuming X_hat and Y_reference are already in log scale

        # Loss function equation
        # L_log(X_hat, Y_reference) = sum(1/2 * X_hat + exp(2*Y_reference - X_hat))
        """
        Custom co-log-likelihood loss function as described in the SAR imaging paper.
        Assumes X_hat and Y_reference are both in log-scale.
    
        Args:
        X_hat (Tensor): Log-transformed predicted pixel values. (X_input -N)
        Y_reference (Tensor): Log-transformed expected noise variance (reference values).
    
        Returns:
        Tensor: The computed co-log-likelihood loss.
        """
        # Ensure the inputs are of the same shape
        if X_hat.shape != Y_reference.shape:
            raise ValueError("X_hat and Y_reference must be of the same shape")

        loss = torch.mean(0.5 * X_hat + (Y_reference/X_hat))

        return loss

    def co_log_likelihood_loss_v2(self, X_hat, Y_reference):

        # Assuming X_hat and Y_reference are already in log scale

        # Loss function equation
        # L_log(X_hat, Y_reference) = sum(1/2 * X_hat + exp(2*Y_reference - X_hat))
        """
        Custom co-log-likelihood loss function as described in the SAR imaging paper.
        Assumes X_hat and Y_reference are both in log-scale.
2
        Args:
        X_hat (Tensor): Log-transformed predicted pixel values. (X_input -N)
        Y_reference (Tensor): Log-transformed expected noise variance (reference values).

        Returns:
        Tensor: The computed co-log-likelihood loss.
        """
        # Ensure the inputs are of the same shape
        if X_hat.shape != Y_reference.shape:
            raise ValueError("X_hat and Y_reference must be of the same shape")


        # import matplotlib.pyplot as plt
        # import numpy as np
        # Ynpy = Y_reference[0,0,:,:].cpu().numpy()
        # Xnpy = X_hat[0,0,:,:].cpu().numpy()
        #
        # plt.imshow(np.exp(Ynpy)**(1/2), cmap="gray", vmax=800)
        # plt.figure()
        # plt.imshow(np.exp(Xnpy)**(1/2), cmap="gray", vmax=800)
        # plt.show()
        #
        # ratio = np.divide(np.exp(Xnpy) + 1e-10, (np.exp(Ynpy)) + 1e-10)
        # ratio = np.clip(ratio, 0, 4)
        # plt.figure()
        # plt.imshow(ratio, cmap="gray", vmax=4)
        # plt.show()
        # loss = torch.mean(X_hat + torch.exp(Y_reference - X_hat))

        # loss = torch.sum(X_hat - Y_reference + torch.exp(Y_reference - X_hat))
        loss = torch.sum(X_hat - Y_reference + torch.exp(Y_reference - X_hat))

        return loss

    def JD_div(self, X_hat, Y_reference):
        
        log_data_X = X_hat * (Max- min) + min
        log_data_Y = Y_reference * (Max- min) + min
        
        
        
        p = F.softmax(log_data_X, dim = 1 )
        q = F.softmax (log_data_Y, dim = 1)
        m = 0.5 * (p + q)

        kl_div_p_m = F.kl_div(m, p, reduction='batchmean')
        kl_div_q_m = F.kl_div(m, q, reduction='batchmean')

        js_divergence = 0.5 * (kl_div_p_m + kl_div_q_m)
        return js_divergence
        
       

