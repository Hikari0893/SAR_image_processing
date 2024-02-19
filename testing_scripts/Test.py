from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # '1,3,4,5,6,7' for 12, '0','1','2','3' on 21

from cnn_despeckling.Jarvis import *
from cnn_despeckling.utils import (store_data_and_plot,
                                   reconstruct_image_from_processed_patches,
                                   process_patches_with_model)

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

# Field
patch_index = 200

# Part of city
# patch_index = 1300

# Instantiate the Model
model = Autoencoder_Wilson_Ver1.load_from_checkpoint(checkpoint)  


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
directory =str('../')
# _________________________________________________________________________________________
model.to(device)

model_result_folder = os.path.join("../results/", Path(checkpoint).stem)
if not os.path.exists(model_result_folder):
    print(f"CreateFolder: {model_result_folder}")
    os.makedirs(model_result_folder)

patches2process = [400,1100]
if patch_index is not None:
    for patch_index in patches2process:
        ORIGINAL_DIMS = (256, 256)

        full_patches = "../data/patches/General_SLC/"
        org_slc_patches = [f for f in os.listdir(full_patches) if f.endswith('.npy')]
        org_slc_patches.sort(key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))  # Sorting by index
        p_index = [org_slc_patches[patch_index]]


        noisy_slc = np.squeeze(np.load(os.path.join(full_patches, p_index[0])))
        amp_full = np.sqrt(noisy_slc)
        noisy_threshold = 3 * np.mean(amp_full)  # + 3 * np.std(reconstructed_image_A)
        filename = model_result_folder + '/test_full_noisy_patchnum_'+str(patch_index)
        store_data_and_plot(amp_full, noisy_threshold, filename)

        sublook_a = np.squeeze(np.load(os.path.join(patch_folder_A, p_index[0])))
        int_a = sublook_a
        sublook_a = np.sqrt(sublook_a)
        filename = model_result_folder + '/test_sublookA_noisy_patchnum_'+str(patch_index)
        store_data_and_plot(sublook_a, noisy_threshold, filename)

        processed_patchesA = process_patches_with_model(patch_folder_A, model, device, desc='patches SLA',
                                                        patch_index=patch_index)
        reconstructed_image_A = reconstruct_image_from_processed_patches(processed_patchesA, ORIGINAL_DIMS,
                                                                         desc='patches SLA')

        filename = model_result_folder + '/test_sublookA_filtered_patchnum_'+str(patch_index)
        store_data_and_plot(reconstructed_image_A,  noisy_threshold, filename)

        # sublook_b = np.squeeze(np.load(os.path.join(full_patches, p_index[0])))
        # int_b = sublook_b
        # sublook_b = np.sqrt(sublook_b)
        # filename = '../results/test_sublookB_noisy_patchnum_'+str(patch_index)
        # store_data_and_plot(sublook_b, noisy_threshold, filename)

        patch_folder_B = full_patches
        processed_patchesB = process_patches_with_model(patch_folder_B, model, device, desc='patches SLB',
                                                        patch_index=patch_index)
        reconstructed_image_B = reconstruct_image_from_processed_patches(processed_patchesB, ORIGINAL_DIMS,
                                                                         desc='patches SLB')
        filename = model_result_folder + '/test_full_filtered_patchnum_'+str(patch_index)
        # 3 * np.mean(reconstructed_image_B)
        store_data_and_plot(reconstructed_image_B,  noisy_threshold, filename)

        # sum = (reconstructed_image_A + reconstructed_image_B)/2
        # filename = '../results/test_AB_patchnum'+str(patch_index)
        # store_data_and_plot(sum,  noisy_threshold, filename)

else:
    ORIGINAL_DIMS = (8244,9090)
    # Plotting original and filtered amplitude images
    processed_patchesA = process_patches_with_model(patch_folder_A, model, device, desc='patches SLA', patch_index=patch_index)
    reconstructed_image_A = reconstruct_image_from_processed_patches(processed_patchesA, ORIGINAL_DIMS, desc='patches SLA')
    threshold = 3*np.mean(reconstructed_image_A)# + 3 * np.std(reconstructed_image_A)
    filename = '../test_sublookA_filtered'
    store_data_and_plot(reconstructed_image_A, threshold, filename)

    processed_patchesB = process_patches_with_model(patch_folder_B, model, device, desc='patches SLB', patch_index=patch_index)
    reconstructed_image_B = reconstruct_image_from_processed_patches(processed_patchesB, ORIGINAL_DIMS, desc='patches SLB')
    threshold = 3*np.mean(reconstructed_image_B)# + 3 * np.std(reconstructed_image_B)
    filename = '../test_sublookB_filtered'
    store_data_and_plot(reconstructed_image_B, threshold, filename)

# SLC = (reconstructed_image_A + reconstructed_image_B)/2
# threshold = np.mean(SLC) + 3 * np.std(SLC)
# filename = '/path/'
# store_data_and_plot(SLC, threshold, filename)


# ENL:
# image_sar =SLC[:,:]   # Placeholder for SAR image.
# mask = np.ones((1000, 1000), dtype=bool)  # Placeholder for mask.

# enl = calculate_enl(image_sar, mask)
# print(f"ENL: {enl}")

 
