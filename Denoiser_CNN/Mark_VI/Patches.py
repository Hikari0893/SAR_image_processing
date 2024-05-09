import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import glob


def load_sar_images(filelist, num_channels=2):
    """
    Load SAR images from files.

    Args:
        filelist (str or list of str): Path to the file or list of file paths.
        num_channels (int): Number of channels in the image.

    Returns:
        numpy.ndarray or list of numpy.ndarray: Loaded image(s).
    """

    def load_image(file):
        try:
            image = np.load(file)
            if image.ndim != 3 or image.shape[-1] != num_channels:
                raise ValueError("Image format is incorrect")
            return image.reshape(1, image.shape[0], image.shape[1], num_channels)
        except Exception as e:
            print(f"Error loading {file}: {e}")
            return None

    if isinstance(filelist, str):
        return load_image(filelist)

    return [load_image(file) for file in filelist if load_image(file) is not None]


def test(sublooks, stride, output_base_folder, pat_size=256):
    if not os.path.exists(output_base_folder):
        os.makedirs(output_base_folder)

    GeneralNames = ["General_A", "General_B", "General_SLC"]

    for idx, sublook in enumerate(sublooks):
        # Create a folder for this image's patches GeneralX
        image_output_folder = os.path.join(output_base_folder, GeneralNames[idx])
        if not os.path.exists(image_output_folder):
            print(f"CreateFolder: {image_output_folder}")
            os.makedirs(image_output_folder)

        print(f"Sublook: {image_output_folder}")
        patch_count = 0

        for file in sublook:

            SL = load_sar_images(file)
            if SL is None:
                print(f"Failed to load image from {file}. Skipping this file.")
                continue
            SL = SL.astype(np.float32)
            if SL.shape[-1] == 2:
                SL = (np.abs(SL[:, :, :, 0] + 1j * SL[:, :, :, 1])) ** 2
            SL_reshaped = SL.reshape(SL.shape[0], SL.shape[1], SL.shape[2])
            im_h, im_w = SL.shape[1:]

            x_range = range(0, max(im_h - pat_size, 1), stride)
            y_range = range(0, max(im_w - pat_size, 1), stride)

            for x in x_range:
                for y in y_range:
                    patch = SL_reshaped[:, x:x + pat_size, y:y + pat_size]

                    patch_file = os.path.join(image_output_folder, f'patch_{patch_count:07}.npy')
                    np.save(patch_file, patch)
                    patch_count += 1
            print(patch_count)


def visualize_patches(folder, num_patches=5):
    """
    Visualizes a specified number of patches from a given folder.

    :param folder: Path to the folder containing patch files.
    :param num_patches: Number of patches to visualize.
    """
    patch_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.npy')]
    patch_files = patch_files[:num_patches]  # Select the first 'num_patches' files

    plt.figure(figsize=(15, 3))
    for i, patch_file in enumerate(patch_files, 1):
        patch = np.squeeze(np.load(patch_file))
        plt.subplot(1, num_patches, i)
        plt.imshow(patch, cmap='gray')
        plt.axis('off')
        plt.title(f'Patch {i}')
    plt.show()

stride = 256

output_base_folder = '../Dayana/Dataset/training_patches/'
A_files = sorted(glob.glob('../Dayana/training/sublookA*'))
B_files = sorted(glob.glob('../Dayana/training/sublookB*'))
slc_files = sorted(glob.glob('../Dayana/training/TESTING_/tsx*'))
test([A_files, B_files, slc_files], stride, output_base_folder)