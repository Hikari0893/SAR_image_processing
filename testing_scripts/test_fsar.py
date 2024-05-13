from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # '1,3,4,5,6,7' for 12, '0','1','2','3' on 21
from cv2 import imwrite

from cnn_despeckling.Jarvis import *
from cnn_despeckling.utils import (plot_residues,
                                   threshold_and_clip,
                                   mem_process_patches_with_model,
                                   create_patches_n, assemble_patches,
                                   T_EF)
from cnn_despeckling.ste_io import rrat

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

stride = 256
ovr = 128

rg_sta = 8000
rg_end = rg_sta + 256 * 7
az_sta = 2000
az_end = az_sta + 256 * 7
crop1 = [rg_sta, rg_end, az_sta, az_end]

rg_sta = 10000
rg_end = rg_sta + 256 * 3
az_sta = 2000
az_end = az_sta + 256 * 3
crop2 = [rg_sta, rg_end, az_sta, az_end]

crops = [crop1, crop2]

dir = "../cnn_despeckling/model_checkpoints/"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

crops = [[1000, 1000 + 256*22, 1000, 1000 + 256*22]]
# crops = [[1000, 1000 + 256*2, 1000, 1000 + 256*2], [1000, 1000 + 256*6, 1000, 1000 + 256*6]]

crops = [[0, 0, 0, 0]]


tsx = False

models = ['WilsonVer1_Net_mse_leaky_relu_10_30_0.001_az_fsar_dinsar.ckpt']
sub_path = "/ste/usr/amao_jo/estudiantes/dayana/SAR_image_processing/data/fsar/az_sublookA_slc_22dinsar0202_Xvv_t01X.npy"
# sub_path = "/ste/usr/amao_jo/estudiantes/dayana/SAR_image_processing/data/fsar/rg_sublookA_slc_22dinsar0202_Xvv_t01X.npy"

slc_path1 = "/ste/img/22OP22AF/FL01/PS14/T01X_sla/RGI/RGI-SR/slc_22op22af0114_Xvv_t01X_sla.rat"
slc_path2 = "/ste/img/22DINSAR/FL02/PS04/T01X/RGI/RGI-SR/slc_22dinsar0204_Xvv_t01X.rat"

data2filter = [slc_path2, sub_path, slc_path1]

data2filter = [slc_path2]

for checkpoint in models:
    for path in data2filter:
        for crop in crops:
            # Instantiate the Model
            model = Autoencoder_Wilson_Ver1.load_from_checkpoint(dir + checkpoint)
            model.to(device)
            rg_sta, rg_end, az_sta, az_end = crop

            # rg_sta, rg_end, az_sta, az_end = crop
            model_result_folder = os.path.join("../results/", Path(checkpoint).stem + "/" + Path(path).stem)
            if not os.path.exists(model_result_folder):
                print(f"CreateFolder: {model_result_folder}")
                os.makedirs(model_result_folder)

            # arr = np.load(path)
            # arr = arr[rg_sta:rg_end, az_sta:az_end, 0] + 1j * arr[rg_sta:rg_end, az_sta:az_end, 1]

            if path == sub_path:
                arr = np.load(path)
                arr = arr[rg_sta:rg_end, az_sta:az_end, 0] + 1j * arr[rg_sta:rg_end, az_sta:az_end, 1]
                arr = np.abs(arr)**2
            else:
                # arr = rrat(path, block=[0, 256*78, 0, 256*23])
                if crop == [0, 0, 0, 0]:
                    arr = rrat(path)
                else:
                    arr = rrat(path, block=[rg_sta, rg_end, az_sta, az_end])
                arr = np.abs(arr)**2

            patches, r1, r2 = create_patches_n(arr[np.newaxis,...], ovr=ovr)
            patch_list = mem_process_patches_with_model(patches, model, device, tsx=tsx)
            clean_int = assemble_patches(patch_list, r1, r2, ovr=ovr)
            clean_amp = np.sqrt(clean_int)
            orig_amp = np.sqrt(arr)

            noisy2plot = np.flipud(orig_amp)
            filename = model_result_folder + '/noisy' + str(crop) + '.png'
            imwrite(filename, threshold_and_clip(noisy2plot, noisy2plot))

            amp2plot = np.flipud(clean_amp)
            filename = model_result_folder + '/clean' + str(crop) + '.png'
            imwrite(filename, threshold_and_clip(noisy2plot, amp2plot))

            # Residuals

            try:
                noisy2plot + amp2plot
            except ValueError:
                print("Input and output have different dimensions. Skipping residuals quicklooks!")
            else:
                res_sl = plot_residues(noisy2plot ** 2, amp2plot ** 2, res_only=True)
                filename = model_result_folder + '/residual' + str(crop) + '.png'
                imwrite(filename, T_EF(res_sl, 0.5, 1.5, 0, 255))


            if crop == crop1:
                green_crop = [1200, 1400, 1200, 1400]
                res_sl = plot_residues(orig_amp ** 2, clean_amp ** 2, res_only=True)
                patch_int = clean_int[green_crop[0]:green_crop[1], green_crop[2]:green_crop[3]]
                enl_clean = (np.mean(patch_int) ** 2) / (np.std(patch_int) ** 2)

                print("Approach MoR: " + str(np.mean(res_sl[green_crop[0]:green_crop[1], green_crop[2]:green_crop[3]])) +
                      ", VoR: " + str(np.var(res_sl[green_crop[0]:green_crop[1], green_crop[2]:green_crop[3]])))
                print("Approach ENL: " + str(enl_clean))
