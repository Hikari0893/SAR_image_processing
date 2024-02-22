import os
import sys
import numpy as np
import torch
import torch.nn as nn
import glob
from torch.utils.data import Dataset, DataLoader
import logging

import json

# Logging Configuration to log errors in a file
logging.basicConfig(filename='dataset_errors.log', level=logging.ERROR, 
                    format='%(asctime)s %(levelname)s:%(message)s')

main_path = os.path.dirname(os.path.abspath(__file__))[:-4]
sys.path.insert(0, main_path+"patches_folder")
sys.path.insert(0, main_path+"Dataset")
sys.path.insert(0, main_path+"Data")
sys.path.insert(0, main_path+"Preprocessing__")

# Load global parameters from JSON file
config_file = "../config.json"  # Update with your JSON file path

with open(config_file, 'r') as json_file:
    global_parameters = json.load(json_file)

only_test = bool(global_parameters['global_parameters']['ONLYTEST'])

class NPYDataset(Dataset):
    def __init__(self, npy_folder, ratio=1):
        self.npy_folder = npy_folder
        self.filenames = sorted([f for f in os.listdir(npy_folder) if f.endswith('.npy')])
        if ratio != 1:
            num_files_to_select = int(ratio * len(self.filenames))
            self.filenames = self.filenames[:num_files_to_select]

    def __len__(self): #to return the total number of samples in the dataset
        return len(self.filenames)

    def __getitem__(self, idx): #to load and return a specific sample per index
        filepath = os.path.join(self.npy_folder, self.filenames[idx])
        
        #if an error occurs during loading of a .npy file, it is caught, and a message is printed
        try:  
            #data = np.squeeze(np.load(filepath),axis=-1)
            data = np.load(filepath)
        except Exception as e:
            logging.error(f"Error loading {filepath}: {e}")
            print(f"Error loading {filepath}: {e}")
            return torch.zeros(1)  # Or handle the error in an appropriate way for your use case
        
        data = torch.from_numpy(data)
        
        return data


class NPYDataset_Mem(Dataset):
    def __init__(self, data_folder, patch_size, pattern):
        self.data_folder = data_folder
        self.patch_size = patch_size
        self.training_data = sorted(glob.glob(data_folder + pattern))
        self.data = self._load_data()

    def _load_data(self):
        data = []
        for filename in self.training_data:
            try:
                # Load the .npy file
                loaded_data = np.load(filename)
                intensity = (np.abs(loaded_data[:, :, 0] + 1j * loaded_data[:, :, 1])) ** 2
                # Create patches from the loaded data
                patches = self._create_patches(intensity[np.newaxis,:])
                data.extend(patches)
            except Exception as e:
                logging.error(f"Error loading {filename}: {e}")
                print(f"Error loading {filename}: {e}")

        return data

    def _create_patches(self, data):
        n = self.patch_size
        ovr = 0
        assert n > ovr >= 0
        s1, s2 = np.shape(data[0,...])
        r1, r2 = (s1 - n) // (n - ovr) + 1, (s2 - n) // (n - ovr) + 1  # number of patches per column/row
        patch_list = []
        for i in range(r1):
            for j in range(r2):
                patch = data[:, i * (n - ovr):i * (n - ovr) + n, j * (n - ovr):j * (n - ovr) + n]
                patch_list.append(patch)
        return patch_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
class CombinedDataset(Dataset):
    def __init__(self, dataset_A, dataset_B):
        self.dataset_A = dataset_A
        self.dataset_B = dataset_B
        assert len(self.dataset_A) == len(self.dataset_B), "Datasets must be the same size"

    def __len__(self):
        return len(self.dataset_A)

    def __getitem__(self, idx):
        data_A = self.dataset_A[idx]
        data_B = self.dataset_B[idx]
        return data_A, data_B
    
    

# Parallel Data Loading, adjust this number based on your system's capabilities
# The "only_test" mode, only one DataLoader is created for the test set.
    
class NPYDataLoader:
    def __init__(self, batch_size, num_workers=8, folder_A =None, folder_B = None, only_test=only_test,
                 data_folder = None, patternA = None, patternB = None):
        if data_folder is not None:
            self.dataset_A = NPYDataset_Mem(data_folder, 256, patternA)
            self.dataset_B = NPYDataset_Mem(data_folder, 256, patternB)
        else:
            self.dataset_A = NPYDataset(folder_A)
            self.dataset_B = NPYDataset(folder_B)
        if only_test == False:
                
            training_ratio   = 0.75
            validation_ratio = 0.25
            test_ratio       = 0
            
            dataset_size    = len(self.dataset_A)
            training_size   = int(training_ratio * dataset_size)
            validation_size = int(validation_ratio * dataset_size)
            test_size       = dataset_size - training_size - validation_size
            
            # Split indices instead of datasets directly
            indices = torch.randperm(dataset_size).tolist()
            train_indices = indices[:training_size]
            val_indices   = indices[training_size:training_size + validation_size]
            test_indices  = indices[training_size + validation_size:]
            
            train_set_A = torch.utils.data.Subset(self.dataset_A, train_indices)
            train_set_B = torch.utils.data.Subset(self.dataset_B, train_indices)
            val_set_A   = torch.utils.data.Subset(self.dataset_A, val_indices)
            val_set_B   = torch.utils.data.Subset(self.dataset_B, val_indices)
            test_set_A  = torch.utils.data.Subset(self.dataset_A, test_indices)
            test_set_B  = torch.utils.data.Subset(self.dataset_B, test_indices)
            
            self.train_loader = DataLoader(CombinedDataset(train_set_A, train_set_B), batch_size=batch_size, shuffle=True, num_workers=num_workers)
            self.val_loader   = DataLoader(CombinedDataset(val_set_A, val_set_B), batch_size=batch_size, shuffle=True, num_workers=num_workers)
            self.test_loader  = DataLoader(CombinedDataset(test_set_A, test_set_B), batch_size=batch_size, shuffle=False, num_workers=num_workers)
        else:
            self.test_loader  = DataLoader(CombinedDataset(self.dataset_A, self.dataset_B), batch_size=100, shuffle=False, num_workers=16)


