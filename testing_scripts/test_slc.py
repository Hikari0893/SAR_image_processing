from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # '1,3,4,5,6,7' for 12, '0','1','2','3' on 21
from cv2 import imwrite

from cnn_despeckling.Jarvis import *
from cnn_despeckling.utils import (plot_residues,
                                   threshold_and_clip,
                                   mem_process_patches_with_model,
                                   create_patches_n, assemble_patches,
                                   T_EF)

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

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
ovr = 128

subB_path = "../data/testing/sublookB_neustrelitz.npy"
slc_path = "../data/testing/tsx_hh_slc_neustrelitz.npy"
id = "neustrelitz"

slc_path = "../data/testing/tsx_hh_slc_hamburg.npy"
id = "hamburg"

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

# Debug crop
rg_sta = 10000 + 256*3
rg_end = rg_sta + 256
az_sta = 8000 + 256*2
az_end = az_sta + 256
crop3 = [rg_sta, rg_end, az_sta, az_end]

# data2filter = [subB_path, slc_path]
crops = [crop1, crop2]
#
# crops = [crop3]
data2filter = [slc_path]
for path in data2filter:
    for crop in crops:
        rg_sta, rg_end, az_sta, az_end = crop
        model_result_folder = os.path.join("../results/", Path(checkpoint).stem + "/" + Path(path).stem)
        if not os.path.exists(model_result_folder):
            print(f"CreateFolder: {model_result_folder}")
            os.makedirs(model_result_folder)

        arr = np.load(path)
        arr = arr[rg_sta:rg_end, az_sta:az_end, 0] + 1j * arr[rg_sta:rg_end, az_sta:az_end, 1]
        arr = np.abs(arr)
        arr = arr**2

        patches = create_patches_n(arr[np.newaxis,...], ovr=ovr)
        patch_list = mem_process_patches_with_model(patches, model, device)
        clean_int = assemble_patches(patch_list, ovr=ovr)
        clean_amp = np.sqrt(clean_int)
        orig_amp = np.sqrt(arr)

        noisy2plot = np.flipud(orig_amp)
        filename = model_result_folder + '/noisy' + str(crop) + '.png'
        imwrite(filename, threshold_and_clip(noisy2plot, noisy2plot))

        amp2plot = np.flipud(clean_amp)
        filename = model_result_folder + '/clean' + str(crop) + '.png'
        imwrite(filename, threshold_and_clip(noisy2plot, amp2plot))

        # Residuals
        res_sl = plot_residues(noisy2plot ** 2, amp2plot ** 2, res_only=True)
        filename = model_result_folder + '/residual' + str(crop) + '.png'
        imwrite(filename, T_EF(res_sl, 0.5, 1.5, 0, 255))

        if id == "neustrelitz" and crop == crop1:
            green_crop = [1330, 1410, 330, 400]
            red_crop = [320, 520, 360, 560]
            # Create figure and axes
            fig, ax = plt.subplots(figsize=(15, 15))
            ax.imshow(threshold_and_clip(noisy2plot, noisy2plot), cmap="gray")
            ax.add_patch(Rectangle((green_crop[2], orig_amp.shape[0] - green_crop[0] - 100),
                                   100, 100,
                                   fc='none',
                                   ec='g',
                                   lw=3))
            ax.add_patch(Rectangle((red_crop[2], orig_amp.shape[0] - red_crop[0] - 200),
                                   200, 200,
                                   fc='none',
                                   ec='r',
                                   lw=3))
            plt.axis("off")
            filename = model_result_folder + '/crops_colored.png'
            plt.savefig(filename, bbox_inches='tight', pad_inches=0, transparent=True)

            # Calculating Mean of Ratio (MoR) and variance of ratio (VoR) for homogeneous region (green crop)
            patch_int = clean_int[green_crop[0]:green_crop[1], green_crop[2]:green_crop[3]]
            enl_clean = (np.mean(patch_int) ** 2) / (np.std(patch_int) ** 2)

            print("-----")
            # A mean of ratio (MoR) significantly different from one indicates, again, some radiometric distortion.
            # Assuming MoR ~= 1, the variance of ratio (VoR) provides insight about under/oversmoothing phenomena.
            # A VoR < 1 indicates undersmoothing, that is, part of the speckle remains in the filtered image,
            # whereas VoR > 1 indicates oversmoothing, that is, the filter eliminates also some details of the
            # underlying image.
            # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6515372&tag=1
            print("Approach MoR: " + str(np.mean(res_sl[green_crop[0]:green_crop[1], green_crop[2]:green_crop[3]])) +
                  ", VoR: " + str(np.var(res_sl[green_crop[0]:green_crop[1], green_crop[2]:green_crop[3]])))
            print("Approach ENL: " + str(enl_clean))

            zoom_noisy = np.flipud(orig_amp[red_crop[0]:red_crop[1], red_crop[2]:red_crop[3]])
            zoom_pred = np.flipud(clean_amp[red_crop[0]:red_crop[1], red_crop[2]:red_crop[3]])
            imwrite(model_result_folder + '/zoom_corner_reflectors_noisy.png', threshold_and_clip(noisy2plot, zoom_noisy))
            imwrite(model_result_folder + '/zoom_corner_reflectors_clean.png', threshold_and_clip(noisy2plot, zoom_pred))
