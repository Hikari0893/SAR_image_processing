import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import logging

# Logging Configuration
logging.basicConfig(filename='dataset_errors.log', level=logging.ERROR, 
                    format='%(asctime)s %(levelname)s:%(message)s')

main_path = os.path.dirname(os.path.abspath(__file__))[:-4]
sys.path.insert(0, main_path+"patches_folder")
sys.path.insert(0, main_path+"Dataset")
sys.path.insert(0, main_path+"Data")
sys.path.insert(0, main_path+"Preprocessing__")

npy_folder = "/home/tonix/Documents/Dayana/patches_folder"
#npy_folder = "/home/tonix/Documents/Dayana/Dataset/sublookB"

class NPYDataset(Dataset):
    def __init__(self):
        self.npy_folder = npy_folder
        self.filenames = [f for f in os.listdir(npy_folder) if f.endswith('.npy')]
        
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filepath = os.path.join(self.npy_folder, self.filenames[idx])
        
        #if an error occurs during loading of a .npy file, it is caught, and a message is printed
        try:  
            #data = np.squeeze(np.load(filepath),axis=-1)
            data = np.load(filepath)
        except Exception as e:
            logging.error(f"Error loading {filepath}: {e}")
            print(f"Error loading {filepath}: {e}")
            return torch.zeros(1)  # Or handle the error in an appropriate way for your use case
        
        data = torch.from_numpy(data).float()
        
        return data

class NPYDataLoader(object):
    # Parallel Data Loading, adjust this number based on your system's capabilities
    
    def __init__(self, batch_size, num_workers=8):
          
        self.dataset = NPYDataset()
        # Split ratios (training, validation, testing)
        training_ratio   = 0.6
        validation_ratio = 0.2
        test_ratio       = 0.2  # This ensures the ratios add up to 1

        training_size   = int(training_ratio * len(self.dataset))
        validation_size = int(validation_ratio * len(self.dataset))
        test_size       = len(self.dataset) - training_size - validation_size
        
        # Set the seed for reproducibility
        torch.manual_seed(0)
        train_set, val_set, test_set = random_split(self.dataset, [training_size, validation_size, test_size])

        self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Usage


# Create an instance of the dataset with min_val and max_val

npy_loader = NPYDataLoader(batch_size=32, num_workers=8)

test_0 = True
if test_0:
    def test_dataloader(dataloader):
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader), desc="Processing Batches"):
            # Basic checks like shape of the data
            print(f"Batch {i}, Data Shape: {data.shape}")

            # Check for NaN values
            if torch.isnan(data).any():
                print(f"Warning: NaN values detected in batch {i}")
        
            # Check for infinite values    
            if torch.isinf(data).any():
                print(f"Warning: Infinite values detected in batch {i}")    

            # Count the number of zeros
            num_zeros = torch.sum(data == 0).item()
            print(f"Number of zeros in batch {i}: {num_zeros}")

    # Add any additional checks or processing here
    # Assuming npy_loader is an instance of NPYDataLoader
    print("Testing Train DataLoader")
    test_dataloader(npy_loader.train_loader)

    print("Testing Validation DataLoader")
    test_dataloader(npy_loader.val_loader)

    print("Testing Test DataLoader")
    test_dataloader(npy_loader.test_loader)