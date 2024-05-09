import numpy as np
from scipy.signal import correlate2d
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# Cargamos o definimos los sublooks A y B como arrays 2D
sublook_a = np.load('/home/tonix/Documents/Dayana/sublookA_a013A.npy')# ... tu array 2D para el sublook A
sublook_b = np.load('/home/tonix/Documents/Dayana/sublookB_a013B.npy')# ... tu array 2D para el sublook B

complex_a = sublook_a[:, :, 0] + 1j * sublook_a[:, :, 1]
complex_b = sublook_b[:, :, 0] + 1j * sublook_b[:, :, 1]
del sublook_a, sublook_b

magnitude_a = np.abs(complex_a)
magnitude_a = (magnitude_a - np.min(magnitude_a))/(np.max(magnitude_a) - np.min(magnitude_a))

magnitude_b = np.abs(complex_b)
magnitude_b = (magnitude_b - np.min(magnitude_b))/(np.max(magnitude_b) - np.min(magnitude_b))

del complex_a, complex_b

correlacion = np.corrcoef(magnitude_a, magnitude_b)[0, 1]
print(f"Coeficiente de correlación de Pearson: {correlacion}")

coef_spearman, _ = spearmanr(magnitude_a, magnitude_b)
print(f"Coeficiente de correlación de rango de Spearman: {coef_spearman}")

magnitude_correlation = correlate2d(magnitude_a, magnitude_b, boundary='fill', mode='same')
#phase_correlation = correlate2d(phase_a, phase_b, boundary='fill', mode='same')

# Aplicar la normalización min-max a la correlación
min_val = np.min(magnitude_correlation)
max_val = np.max(magnitude_correlation)
normalized_magnitude_correlation = (magnitude_correlation - min_val) / (max_val - min_val) if (max_val - min_val) else magnitude_correlation



# Aplanar las magnitudes y fases en vectores (1D)
flat_magnitude_a = magnitude_a.ravel()
##flat_phase_a = phase_a.ravel()

flat_magnitude_b = magnitude_b.ravel()
##flat_phase_b = phase_b.ravel()

# Asegurarse de que la cantidad de puntos es la correcta
n_points_a = flat_magnitude_a.shape[0]
n_points_b = flat_magnitude_b.shape[0]
# Concatenar las magnitudes y fases
combined_magnitude = np.stack((flat_magnitude_a, flat_magnitude_b), axis=1)
##combined_phase = np.stack((flat_phase_a, flat_phase_b), axis=1)

# Aplicar PCA a las magnitudes
pca_magnitude = PCA(n_components=2)
pca_magnitude.fit(combined_magnitude)


# Transformar los datos al espacio de componentes principales
transformed_magnitude = pca_magnitude.transform(combined_magnitude)

##transformed_phase = pca_phase.transform(combined_phase)

# Visualizar los componentes principales
plt.figure(figsize=(12, 6))
plt.plot()
# Trama para la magnitud de sublook A
plt.scatter(transformed_magnitude[:n_points_a, 0], transformed_magnitude[:n_points_a, 1],color='cyan', alpha=0.5, label='Sublook A')
plt.title('PCA de Magnitud de Sublooks A y B')
plt.xlabel('Primer Componente Principal')
plt.ylabel('Segundo Componente Principal')
plt.legend()
plt.grid(True)


plt.tight_layout()
plt.show()

# Explicar la varianza
print(f'Varianza explicada por cada componente (Magnitud): {pca_magnitude.explained_variance_ratio_}')
#print(f'Varianza explicada por cada componente (Fase): {pca_phase.explained_variance_ratio_}')

plt.figure(figsize=(12, 6))
plt.plot()
plt.imshow(normalized_magnitude_correlation, cmap='hot', interpolation='nearest')
plt.title('Mapa de Correlación Cruzada')
plt.colorbar()

plt.show()


#Scatter plot con coeficiente de correlación de Pearson
plt.figure(figsize=(8, 6))
plt.scatter(magnitude_a, magnitude_b, label=f'Pearson: {correlacion:.2f}', color='blue', alpha=0.5)

# Scatter plot con coeficiente de correlación de rango de Spearman
plt.scatter(magnitude_a, magnitude_b, label=f'Spearman: {coef_spearman:.2f}', color='red', alpha=0.5)

plt.title('Gráfico de Dispersión con Coeficientes de Correlación')
plt.xlabel('Conjunto 1')
plt.ylabel('Conjunto 2')
plt.legend()
plt.grid(True)
plt.show()

print('-------')