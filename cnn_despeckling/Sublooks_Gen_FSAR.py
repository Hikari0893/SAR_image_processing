import numpy as np
from pathlib import Path
from cnn_despeckling.utils import split_sublooks, plot_sublooks
from cnn_despeckling.ste_io import rrat

rg_sta = 2000
rg_end = 4000
az_sta = 2000
az_end = 4000
crop = [rg_sta, rg_end, az_sta, az_end]

# For FSAR, axis=0 AZIMUTH, axis=1 RANGE (columns = azimuth lines, rows = range lines)
proc_axis = 1
debug = 0

# file_list = ["/ste/img/22OP22AF/FL01/PS14/T01X_sla/RGI/RGI-SR/slc_22op22af0114_Xvv_t01X_sla.rat",
#              "/ste/img/22OP22AF/FL01/PS15/T01X_sla/RGI/RGI-SR/slc_22op22af0115_Xvv_t01X_sla.rat"]

file_list = ["/ste/img/22DINSAR/FL02/PS02/T01X/RGI/RGI-SR/slc_22dinsar0202_Xvv_t01X.rat",
             "/ste/img/22DINSAR/FL02/PS03/T01X/RGI/RGI-SR/slc_22dinsar0203_Xvv_t01X.rat",
             "/ste/img/22DINSAR/FL02/PS05/T01X/RGI/RGI-SR/slc_22dinsar0205_Xvv_t01X.rat",
             "/ste/img/22DINSAR/FL02/PS08/T01X/RGI/RGI-SR/slc_22dinsar0208_Xvv_t01X.rat",
             "/ste/img/22DINSAR/FL02/PS09/T01X/RGI/RGI-SR/slc_22dinsar0209_Xvv_t01X.rat",
             "/ste/img/22DINSAR/FL02/PS10/T01X/RGI/RGI-SR/slc_22dinsar0210_Xvv_t01X.rat"
             ]


campaign = "23GABONX"
FL = "FL02"
PS = ["04","05","07","08","09"]
BAND = "L"

alpha = 0.54

for proc_axis in [0, 1]:
    if proc_axis == 0:
        procid = 'az'
    else:
        procid = 'rg'
    for file in file_list:
        id = Path(file).stem
        print(f"Processing file {id}...")
        if debug:
            complex_image = rrat(file, block=[rg_sta, rg_end, az_sta, az_end])
        else:
            complex_image = rrat(file)

        spatial_sectA, spatial_sectB = split_sublooks(complex_image, proc_axis, alpha, debug)
        pathA = f"../data/fsar/{procid}_sublookA_{id}.npy"
        np.save(pathA, np.stack((np.real(spatial_sectA), np.imag(spatial_sectA)),
                                                                 axis = 2))
        pathB = f"../data/fsar/{procid}_sublookB_{id}.npy"
        np.save(pathB, np.stack((np.real(spatial_sectB), np.imag(spatial_sectB)),
                                                                 axis = 2))
        if debug:
            plot_sublooks(pathA, pathB, complex_image)

        print(f"Done with file {id}...")
        print("----------")

print("Done with FSAR sublooks")