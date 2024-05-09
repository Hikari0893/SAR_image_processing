import numpy as np


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
            return image.reshape(1, image.shape[0],image.shape[1], num_channels)
        except Exception as e:
            print(f"Error loading {file}: {e}")
            return None

    if isinstance(filelist, str):
        return load_image(filelist)

    return [load_image(file) for file in filelist if load_image(file) is not None]


'''dir = ['/home/tonix/Documents/Dayana/sublookA.npy']
a = load_sar_images(dir)
print (a)'''