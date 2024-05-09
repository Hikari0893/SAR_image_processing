import numpy as np
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go


def load_image(image_path):
    """Load an image from a file as a complex numpy array and return its magnitude."""
    data = np.load(image_path)
    return np.abs(data[:,:,0] + 1j * data[:,:,1])

def calculate_statistics(image):
    """Calculate mean and variance of the given image."""
    return np.mean(image), np.var(image)

def find_min_cv_segment(args):
    """Find the segment of the image with the lowest coefficient of variation."""
    image, y_start, y_end, window_size = args
    min_cv = float('inf')
    min_loc = None
    for y in range(y_start, min(y_end, image.shape[0] - window_size)):
        for x in range(image.shape[1] - window_size):
            segment = image[y:y+window_size, x:x+window_size]
            mean, std = np.mean(segment), np.std(segment)
            cv = std / mean if mean else float('inf')
            if cv < min_cv:
                min_cv = cv
                min_loc = (x, y)
    return min_loc, min_cv

def parallel_find_homogeneous_region(image, window_size):
    """Use parallel processing to find the most homogeneous region in the image."""
    with Pool(cpu_count()) as pool:
        step = max(1, image.shape[0] // cpu_count())
        tasks = [(image, i, min(i + step, image.shape[0]), window_size) for i in range(0, image.shape[0], step)]
        results = pool.map(find_min_cv_segment, tasks)
    return min(results, key=lambda x: x[1])

def calculate_correlation_matrix(img1, img2):
    """ Calculate the correlation matrix between two images. """
    assert img1.shape == img2.shape, "Images must be the same size"
    img1_flat = img1.flatten()
    img2_flat = img2.flatten()
    covariance_matrix = np.cov(img1_flat, img2_flat)
    # Correlation coefficient calculation
    d = np.sqrt(np.diag(covariance_matrix))
    return covariance_matrix / d / d[:, None] 

def plot_covariance_matrix(cov_matrix, title):
    """ Plot the covariance or correlation matrix. """
    
    fig, ax = plt.subplots()
    cax = ax.matshow(cov_matrix, cmap='coolwarm')
    # Adding annotations
    for (i, j), val in np.ndenumerate(cov_matrix):
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='white')
    plt.colorbar(cax)
    plt.title(title)
    plt.xlabel('Variable 1')
    plt.ylabel('Variable 2')
    plt.savefig(title)
    plt.show()
    
    
def pixel_by_pixel_correlation(img1, img2, patch_size=3):
    """
    Calculate the pixel-by-pixel correlation between two images.

    Parameters:
    - img1, img2: Two numpy arrays of the same shape.
    - patch_size: The size of the patch to be used for local correlation (must be odd).

    Returns:
    - A correlation matrix where each element represents the correlation
      between corresponding patches centered at each pixel.
    """
    assert img1.shape == img2.shape, "Images must have the same dimensions."
    assert patch_size % 2 == 1, "Patch size must be odd."

    pad_width = patch_size // 2
    img1_padded = np.pad(img1, pad_width, mode='reflect')
    img2_padded = np.pad(img2, pad_width, mode='reflect')

    correlation_matrix = np.zeros(img1.shape)

    for i in range(pad_width, img1_padded.shape[0] - pad_width):
        for j in range(pad_width, img1_padded.shape[1] - pad_width):
            patch1 = img1_padded[i-pad_width:i+pad_width+1, j-pad_width:j+pad_width+1]
            patch2 = img2_padded[i-pad_width:i+pad_width+1, j-pad_width:j+pad_width+1]

            # Normalize patches
            patch1 = (patch1 - np.mean(patch1)) / np.std(patch1)
            patch2 = (patch2 - np.mean(patch2)) / np.std(patch2)

            # Calculate correlation
            correlation = np.mean(patch1 * patch2)
            correlation_matrix[i-pad_width, j-pad_width] = correlation

    return correlation_matrix    

def plot_3d_correlation_matrix(correlation_matrix):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Prepare data for 3D plotting
    x = np.arange(correlation_matrix.shape[1])
    y = np.arange(correlation_matrix.shape[0])
    X, Y = np.meshgrid(x, y)
    Z = correlation_matrix

    # Calculate the mean correlation
    mean_correlation = np.mean(correlation_matrix)
    mean_correlation_x = np.mean(correlation_matrix, axis=0)  # Mean along columns
    mean_correlation_y = np.mean(correlation_matrix, axis=1)  # Mean along rows

    # Plot a surface
    surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', edgecolor='none')

    # Plot the mean correlation as a horizontal plane
    ax.plot_surface(X, Y, np.full_like(Z, mean_correlation), color='k', alpha=0.25, zorder=1)

    # Plot mean lines along the x and y dimensions
    ax.plot(x, np.zeros_like(x), mean_correlation_x, color='yellow', marker='o', markersize=5, label='Mean Correlation (X-axis)')
    ax.plot(np.zeros_like(y), y, mean_correlation_y, color='cyan', marker='^', markersize=5, label='Mean Correlation (Y-axis)')

    # Customize the z axis
    ax.set_zlim(np.min(Z), np.max(Z))
    ax.zaxis.set_major_locator(plt.LinearLocator(10))
    ax.zaxis.set_major_formatter(plt.FormatStrFormatter('%.02f'))

    # Add labels and title
    ax.set_xlabel('Pixel X Coordinate')
    ax.set_ylabel('Pixel Y Coordinate')
    ax.set_zlabel('Correlation Value')
    ax.set_title('3D Visualization of Pixel-by-Pixel Correlation')

    # Add a color bar to indicate the correlation values
    fig.colorbar(surf, shrink=0.5, aspect=5)

    # Legend to explain the plots
    ax.legend()

    plt.savefig('Pixel_by_Pixel_Correlation_with_Mean.png')
    plt.show()
    
    
def plot_with_plotly(matrix):
    x = np.arange(matrix.shape[1])
    y = np.arange(matrix.shape[0])
    X, Y = np.meshgrid(x, y)
    Z = matrix

    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
    fig.update_layout(title='3D Correlation Matrix', autosize=True,
                      scene=dict(
                          xaxis_title='Pixel X Coordinate',
                          yaxis_title='Pixel Y Coordinate',
                          zaxis_title='Correlation Value'))
    fig.show()   


# Example usage:
if __name__ == '__main__':
    
    sla = '/home/tonix/Documents/Dayana/Dataset/training_patches/joel_Data/az_sublookA_bangkok.npy'
    slb = '/home/tonix/Documents/Dayana/Dataset/training_patches/joel_Data/az_sublookB_bangkok.npy'
    image_1 = load_image(sla)
    image_2 = load_image(slb)
    window_size = 64
    top_left_corner, min_cv = parallel_find_homogeneous_region(image_1, window_size)
    if top_left_corner is not None:
        crop_1 = image_1[top_left_corner[1]:top_left_corner[1]+window_size, top_left_corner[0]:top_left_corner[0]+window_size]
        crop_2 = image_2[top_left_corner[1]:top_left_corner[1]+window_size, top_left_corner[0]:top_left_corner[0]+window_size]
    
    plot_covariance_matrix(calculate_correlation_matrix(crop_1, crop_2), title = 'matriz_correlacion.png')
    correlation_matrix = pixel_by_pixel_correlation(crop_1, crop_2, patch_size=3) 
    
    print(np.mean(correlation_matrix.flatten()))  
    plot_3d_correlation_matrix(correlation_matrix) 
    plot_with_plotly(correlation_matrix)
    print(f"Lowest CV found: {min_cv} at location {top_left_corner}")
