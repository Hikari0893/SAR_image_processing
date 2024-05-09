import numpy as np

def calculate_enl(image, mask=None):
    """
   Calculates the Equivalent Number of Views (ENL) for a SAR image.

    Args:
    - image: An array of NumPy representing the SAR image.
    - mask: A NumPy boolean array of the same size as `image`, where True indicates the pixels belonging to the homogeneous region.
            the pixels belonging to the homogeneous region.

    Returns:
    - ENL: The calculated ENL value for the specified region.
    """
    if mask is None:
        # Si no se proporciona una máscara, se utiliza toda la imagen.
        region = image.flatten()
    else:
        # Selecciona solo los píxeles de la región homogénea.
        region = image[mask]

    # Calcula el promedio y la varianza de la región homogénea.
    mean = np.mean(region)
    variance = np.var(region)

    # Calcula el ENL.
    enl = (mean ** 2) / variance

    return enl

# ENL:
load = np.load('/home/tonix/Documents/Dayana/sublookA.npy')
slc = load[:,:,0] + 1j*load[:,:,1]
slc = np.abs(slc)**2
image_sar =slc[5000:6000,5000:6000]   # Placeholder for SAR image.
mask = np.ones((1000, 1000), dtype=bool)  # Placeholder for mask.

enl = calculate_enl(image_sar, mask)
print(f"ENL: {enl}")