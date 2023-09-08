import numpy as np
import matplotlib.pyplot as plt


#lectura de los datos
picture = np.load('converted_image.npy')
#plt.imshow(picture[:,:,1])
#plt.show()

# Reconstruir la imagen compleja a partir de sus partes real e imaginaria
complex_image = picture[:, :, 0] + 1j * picture[:, :, 1]
del picture


# Apply fft transform
spectrum_az    = np.fft.fftshift(np.fft.fft(complex_image,axis=0))
del complex_image
"""
#spectrum_range = np.fft.fftshift(np.fft.fft(complex_image,axis=1))
images_cols = complex_image.shape[0]
images_rows = complex_image.shape[1]
del complex_image


# 1. Genera las ventanas de Hamming 1D
hamming_az    = 1/(np.hamming(images_cols))     #azimuth
#hamming_range = 1/(np.hamming(images_rows))  #range

dehamming_az  = spectrum_az*hamming_az[:, np.newaxis]

#dehamming_sr  = spectrum_range@hamming_range
"""


# Usar el módulo (magnitud) de la imagen compleja
#magnitude_image = complex_image/np.max(np.abs(complex_image))
#print(magnitude_image)

# Create the heatmap
plt.imshow(np.log(np.abs(spectrum_az)), cmap='viridis')  
# 'viridis' is just one of many available colormaps
plt.colorbar()  # Add a colorbar to the plot to show the value-to-color mapping

# Optionally, you can add labels or customize the plot further
plt.title('Heatmap Example')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Show the heatmap
plt.show()
print("dummy")

"""
# Flatten the 2D array to a 1D array
flattened_data = np.abs(complex_image).flatten()

# Create a histogram for the entire data
plt.hist(flattened_data, bins=1000, edgecolor='black') 
plt.show()
plt.figure()
plt.imshow(np.abs(magnitude_image))
plt.show()
"""
"""
# 1. Preparación de la Imagen: Escalado a [0, 1]
scaled_image = (magnitude_image - np.min(magnitude_image)) / (np.max(magnitude_image) - np.min(magnitude_image))
#print("scaled_image")
#print(scaled_image)

# 2. Visualización y Guardado
plt.figure()
plt.imshow(scaled_image, cmap='gray')  # Usar colormap "gray" para imágenes SAR
plt.axis('off')  # Ocultar los ejes
plt.savefig('original_image.png', bbox_inches='tight', pad_inches=0)
plt.show()
"""

#----------------------------------------------------------------------------------------------------------------------------
# Reconstruir la versión compleja de la imagen a partir de sus partes real e imaginaria
complex_image = picture[:10000, :10000, 0] + 1j * picture[:10000, :10000, 1]

# Aplicar FFT 2D
#fft_image = np.fft.fftshift(np.fft.fft2(complex_image))
spectrum_az = np.fft.fftshift(np.fft.fft(complex_image,axis=0))
spectrum_range = np.fft.fftshift(np.fft.fft(complex_image,axis=1))
print(spectrum_az)


#Unweighting
#------------------------------------------------
f_az = 2.76500000000000000e3
f_range = 1.50000000000000000e8
TS_F = 1/(2.5*f_az)
#------------------------------------------------
alpha = 6.00000023841857910e-1
#--------------------------------------------------------------------------------------------------------

# 1. Genera las ventanas de Hamming 1D
hamming_az = 1/(np.hamming(complex_image.shape[0]))
hamming_range = 1/(np.hamming(complex_image.shape[1]))

# 2. Crea la ventana de Hamming 2D
hamming_2D = np.outer(hamming_az, hamming_range)

# 3. Multiplica la imagen por la ventana de Hamming 2D
#windowed_image = complex_image * hamming_2D


#----------------------------------------------------------------------------------------------------------
print(hamming_az)
#H_range = 1/(alpha -(1 - alpha)*np.cos(2*np.pi*f_range))
#----------------------------------------------------
Az = spectrum_az*hamming_az
print(Az)
Range = spectrum_range*hamming_range

plt.figure()
plt.imshow(np.imag(picture[:,:,1]),vmax=255,vmin=0)
plt.savefig('Az.png')
plt.show()
#----------------------------------------------------

def generate_sublooks(spectrum, num_sublooks):
    #tamaño del sublook en el eje azimut
    sublook_size=spectrum.shape[0]//num_sublooks
    
    # guardar sublooks en el dominio espacial
    sublooks = []
    
    for i in range(num_sublooks):
        #Extraer el sublook del espectro
        sublook_spectrum = np.zeros_like(spectrum)
        sublook_spectrum[i*sublook_size:(i+1)*sublook_size] = spectrum[i*sublook_size:(i+1)*sublook_size]
        
        # Transformar el sublook al dominio del espacio
        sublook_space = np.fft.ifft2(np.fft.ifftshift(sublook_spectrum))
        sublooks.append(sublook_space)
    
    return sublooks

num_sublooks = 2  # Definir el número de sublooks que desea
sublooks = generate_sublooks(Az, num_sublooks)
plt.plot(np.real(hamming_az))
plt.savefig('espectro_az.png')
plt.show()

#Visualizar los sublooks
fig, axs = plt.subplots(1, num_sublooks, figsize=(15, 5))

for i, ax in enumerate(axs):
    ax.imshow(np.abs(sublooks[i]), cmap='gray')
    ax.set_title(f'Sublook {i+1}')

plt.tight_layout()
plt.savefig('image.png')
plt.show()