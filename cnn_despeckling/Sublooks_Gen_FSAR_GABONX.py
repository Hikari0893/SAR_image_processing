import numpy as np
from pathlib import Path
from cnn_despeckling.utils import split_sublooks, plot_sublooks
from cnn_despeckling.ste_io import rrat
import os

rg_sta = 2000
rg_end = 4000
az_sta = 2000
az_end = 4000
crop = [rg_sta, rg_end, az_sta, az_end]

# For FSAR, axis=0 AZIMUTH, axis=1 RANGE (columns = azimuth lines, rows = range lines)
proc_axis = 0
debug = 0

# campaign = "23GABONX"
# FL = ["02","03","04","07","08","09","10","11","12","14"]
# PS = ["04", "05", "06", "07"]
# BAND = "P"
# POLS = ["vv"]

campaign = "16AFRISR"
# FL = ["01","02","03","04","06","07","08","09","10","11","12","14"]
FL = ["01","02"]#,"03"]

# PS = ["02", "03", "04", "05", "06", "07", "08", "09", "10"]
PS = ["02"]

BAND = "P"
POLS = ["vv"]


# Take every master from each flight, except Pongara

# "/ste/img/"+
# "22DINSAR/FL02/PS02/T01X/RGI/RGI-SR/slc_22dinsar0202_Xvv_t01X.rat

tryn = "P01"
cmp_dir = os.path.join("/ste/img/", campaign.upper())
file_list = [os.path.join(cmp_dir, 'FL%s' % (fls), 'PS%s' % (pss), 'T' + tryn, 'RGI', 'RGI-SR',
                       f"slc_{campaign.lower()}{fls}{pss}_{BAND}{pol}_t{tryn}.rat"
                       ) for fls in FL for pss in PS for pol in POLS]
MM = []
mm = []

existing_files = []

import os
# Check each file in the list and add existing files to the new list
for f in file_list:
    if os.path.isfile(f):
        existing_files.append(f)
alpha = 0.54


file_list = existing_files
for proc_axis in [0]:
    if proc_axis == 0:
        procid = 'az'
    else:
        procid = 'rg'
    for file in file_list:
        id = Path(file).stem
        print(f"Processing file {id}...")
        # if debug:
        #     complex_image = rrat(file, block=[rg_sta, rg_end, az_sta, az_end])
        # else:
        try:
            complex_image = rrat(file)
        except:
            print("SLC not found, skipping...")
            print("-------")
            continue

        logimg = np.log(np.abs(complex_image)**2 + 1e-10)
        MM.append(np.max(logimg))
        mm.append(np.min(logimg))
        if debug:
            print(f"MAX LOG: {np.max(MM)}")
            print(f"MIN LOG: {np.min(mm)}")
            print(f"Mean of all processed MAX LOG: {np.mean(MM)}")
            print(f"Mean of all processed MIN LOG: {np.mean(mm)}")

        spatial_sectA, spatial_sectB = split_sublooks(complex_image, proc_axis, alpha, debug, 5)
        pathA = f"../data/fsar_afrisar/{procid}_sublookA_{id}.npy"
        np.save(pathA, np.stack((np.real(spatial_sectA), np.imag(spatial_sectA)),
                                                                 axis = 2))
        pathB = f"../data/fsar_afrisar/{procid}_sublookB_{id}.npy"
        np.save(pathB, np.stack((np.real(spatial_sectB), np.imag(spatial_sectB)),
                                                                 axis = 2))
        if debug:
            plot_sublooks(pathA, pathB, complex_image)

        print(f"Done with file {id}...")
        print("----------")

print("Done with FSAR sublooks")