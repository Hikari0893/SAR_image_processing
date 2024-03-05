import numpy as np
from cnn_despeckling.utils import split_sublooks, plot_sublooks

rg_sta = 1000
rg_end = 3000
az_sta = 1000
az_end = 3000
crop = [rg_sta, rg_end, az_sta, az_end]

# Testing ids
# id = 'neustrelitz'
# id = 'hamburg'

# Training ids
# id = 'koeln'
# id = 'shenyang'
# id = 'bangkok'
# id = 'enschene'

# Processing parameters
# For TSX, axis=0 AZIMUTH, axis=1 RANGE (columns = azimuth lines, rows = range lines)
proc_axis = 1
debug = 0

ids = ['koeln', 'shenyang', 'bangkok', 'enschene', 'neustrelitz']
for proc_axis in [0, 1]:
    if proc_axis == 0:
        alpha = 0.61
        procid = 'az'
    else:
        alpha = 0.6
        procid = 'rg'
    for id in ids:
        print(f"Processing file {id}...")
        if id == 'neustrelitz':
            picture = np.load('../data/testing/tsx_hh_slc_' + id + '.npy')
        else:
            picture = np.load('../data/training/tsx_hh_slc_'+id+'.npy')

        # Reconstruct the complex image from its real and imaginary parts.
        if debug:
            complex_image = picture[rg_sta:rg_end, az_sta:az_end, 0] + 1j * picture[rg_sta:rg_end, az_sta:az_end, 1]
        else:
            complex_image = picture[:, :, 0] + 1j * picture[: ,: , 1]

        spatial_sectA, spatial_sectB = split_sublooks(complex_image, proc_axis, alpha, debug)

        if id == 'neustrelitz':
            np.save(f"../data/testing/{procid}_sublookA_{id}.npy", np.stack((np.real(spatial_sectA), np.imag(spatial_sectA)),
                                                                          axis=2))
            np.save(f"../data/testing/{procid}_sublookB_{id}.npy", np.stack((np.real(spatial_sectB), np.imag(spatial_sectB)),
                                                                          axis=2))
        else:
            pathA = f"../data/training/{procid}_sublookA_{id}.npy"
            np.save(pathA, np.stack((np.real(spatial_sectA), np.imag(spatial_sectA)),
                                                                          axis=2))
            pathB = f"../data/training/{procid}_sublookB_{id}.npy"
            np.save(pathB, np.stack((np.real(spatial_sectB), np.imag(spatial_sectB)),
                                                                          axis=2))

        if debug:
            plot_sublooks(pathA, pathB, complex_image)

        print("Done with id: " + str(id))
        print("-----")