import numpy as np
import matplotlib.pyplot as plt


# Load SAR image
picture = np.load('converted_image.npy')
#plt.imshow(picture[:,:,1])
#plt.show()

# Reconstruct the complex image from its real and imaginary parts
complex_image = picture[:, :, 0] + 1j * picture[:, :, 1]
del picture

# Use the modulus (magnitude) of the complex image
magnitude_image = complex_image/np.max(np.abs(complex_image))

# Flatten the 2D array to a 1D array
flattened_data = np.abs(complex_image).flatten()

# Create a histogram for the entire data
plt.hist(flattened_data, bins=1000, edgecolor='black') 
plt.show()
plt.figure()
plt.imshow(np.abs(magnitude_image))
plt.show()

# Apply fft transform
#spectrum_range = np.fft.fftshift(np.fft.fft(complex_image,axis=1))
images_cols_az = complex_image.shape[0]
images_rows_range = complex_image.shape[1]
spectrum_az  = np.zeros_like(complex_image)

spectrum_az0 = np.fft.fftshift(np.fft.fft(complex_image[:,:7000]))
spf0 = np.copy(spectrum_az0)
del spectrum_az0

"""spectrum_az1 = np.fft.fftshift(np.fft.fft(complex_image[:,7001:14000]))
spf1 = np.copy(spectrum_az1)
del spectrum_az1

spectrum_az2 = np.fft.fftshift(np.fft.fft(complex_image[:,14001:21000]))
spf2 = np.copy(spectrum_az2)
del spectrum_az2

spectrum_az3 = np.fft.fftshift(np.fft.fft(complex_image[:,21001:28000]))
spf3 = np.copy(spectrum_az3)
del spectrum_az3

spectrum_az4 = np.fft.fftshift(np.fft.fft(complex_image[:,28001:29828]))
spf4 = np.copy(spectrum_az4)
del spectrum_az4

del complex_image"""


"""# Generate 1D Hamming windows

#-------------------Parameters---------------------------
alpha = 6.00000023841857910e-1
f_az = 2.76500000000000000e3
f_range = 1.50000000000000000e8
#---------------------------------------------------------

hamming_windows = alpha -(1 - alpha)*np.cos(2*np.pi*f_az)"""
hamming_az    = 1/(np.hamming(images_cols_az))     #azimuth
#hamming_range = 1/(np.hamming(images_rows))  #range

dehamming_az  = spf0*hamming_az[:, np.newaxis]

plt.figure()
plt.plot(np.real(dehamming_az))  # Usar colormap "gray" para imágenes SAR
plt.axis('off')  # Ocultar los ejes
plt.savefig('odehamming_az.png', bbox_inches='tight', pad_inches=0)
plt.show()

#dehamming_sr  = spectrum_range@hamming_range



#print(magnitude_image)

# Create the heatmap
"""plt.imshow(np.log(np.abs(spectrum_az)), cmap='viridis')  
# 'viridis' is just one of many available colormaps
plt.colorbar()  # Add a colorbar to the plot to show the value-to-color mapping

# Optionally, you can add labels or customize the plot further
plt.title('Heatmap Example')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Show the heatmap
plt.show()
print("dummy")

plt.imshow(dehamming_az)


#Unweighting
#------------------------------------------------
f_az = 2.76500000000000000e3
f_range = 1.50000000000000000e8
TS_F = 1/(2.5*f_az)
#------------------------------------------------
alpha = 6.00000023841857910e-1
#--------------------------------------------------------------------------------------------------------

# 1. Genera las ventanas de Hamming 1D
#hamming_az = 1/(np.hamming(complex_image.shape[0]))
#hamming_range = 1/(np.hamming(complex_image.shape[1]))

# 2. Crea la ventana de Hamming 2D
#hamming_2D = np.outer(hamming_az, hamming_range)

# 3. Multiplica la imagen por la ventana de Hamming 2D
#windowed_image = complex_image * hamming_2D


#----------------------------------------------------------------------------------------------------------
#print(hamming_az)
#H_range = 1/(alpha -(1 - alpha)*np.cos(2*np.pi*f_range))
#----------------------------------------------------
#Az = spectrum_az*hamming_az
#print(Az)
#Range = spectrum_range*hamming_range
"""
#plt.figure()
#plt.imshow(np.imag(picture[:,:,1]),vmax=255,vmin=0)
#plt.savefig('Az.png')
#plt.show()
"""
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
    del spectrum
num_sublooks = 2  # Definir el número de sublooks que desea
sublooks = generate_sublooks(dehamming_az, num_sublooks)
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
plt.show()"""