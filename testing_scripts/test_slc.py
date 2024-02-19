from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # '1,3,4,5,6,7' for 12, '0','1','2','3' on 21

from cnn_despeckling.Jarvis import *
from cnn_despeckling.utils import (store_data_and_plot,
                                   reconstruct_image_from_processed_patches,
                                   mem_process_patches_with_model, create_patches_n, assemble_patches)

config_file = "../config.json"  # Update with your JSON file path

# Load the global parameters from the JSON file
with open(config_file, 'r') as json_file:
    global_parameters = json.load(json_file)

# Access the global parameters

archive = global_parameters['global_parameters']['ARCHIVE']
checkpoint = str(global_parameters['global_parameters']['CKPT'])
patch_folder_A = global_parameters['global_parameters']['INPUT_FOLDER']
patch_folder_B = global_parameters['global_parameters']['REFERENCE_FOLDER']
num_workers = int(global_parameters['global_parameters']['NUMWORKERS'])

# Instantiate the Model
model = Autoencoder_Wilson_Ver1.load_from_checkpoint(checkpoint)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
directory = str('../')
# _________________________________________________________________________________________
model.to(device)


stride = 256
subB_path = "../data/testing/sublookB_neustrelitz.npy"
slc_path = "../data/testing/tsx_hh_slc_neustrelitz.npy"

# City crop
rg_sta = 8000
rg_end = rg_sta + 256 * 7
az_sta = 10000
az_end = az_sta + 256 * 7
crop1 = [rg_sta, rg_end, az_sta, az_end]

# Forest crop
rg_sta = 10000
rg_end = rg_sta + 256 * 3
az_sta = 8000
az_end = az_sta + 256 * 3
crop2 = [rg_sta, rg_end, az_sta, az_end]

data2filter = [subB_path, slc_path]
crops = [crop1, crop2]
for path in data2filter:
    for crop in crops:
        rg_sta, rg_end, az_sta, az_end = crop
        model_result_folder = os.path.join("../results/", Path(checkpoint).stem + "/" + Path(path).stem)
        if not os.path.exists(model_result_folder):
            print(f"CreateFolder: {model_result_folder}")
            os.makedirs(model_result_folder)

        arr = np.load(path)
        arr = arr[rg_sta:rg_end, az_sta:az_end, 0] + 1j * arr[rg_sta:rg_end, az_sta:az_end, 1]

        arr = np.abs(arr)**2
        # vmax = np.mean(arr) * 3
        # plt.imshow(arr, cmap="gray", vmax=vmax)
        ovr = 128

        patches = create_patches_n(arr[np.newaxis,...], ovr=ovr)

        aaa = mem_process_patches_with_model(patches, model, device)

        nb_batch = len(aaa)
        nb_patches_side = round(nb_batch ** 0.5)

        reconstructed_image_A = assemble_patches(aaa, nb_patches_side, nb_patches_side, ovr=ovr)
        # reconstructed_image_A = reconstruct_image_from_processed_patches(aaa, arr.shape, desc='patches SLA', stride=256)
        reconstructed_image_A = np.sqrt(reconstructed_image_A)

        orig_amp = np.sqrt(arr)
        noisy_threshold= np.mean(orig_amp)*3

        filename = model_result_folder + '/noisy' + str(crop)
        # 3 * np.mean(reconstructed_image_B)
        store_data_and_plot(orig_amp, noisy_threshold, filename)

        filename = model_result_folder + '/clean' + str(crop)
        # 3 * np.mean(reconstructed_image_B)
        store_data_and_plot(reconstructed_image_A, noisy_threshold, filename)
