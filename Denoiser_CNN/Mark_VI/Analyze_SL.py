import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def plot_histogram_and_pdf(data, title, color):
    # Flatten the data to 1D and calculate intensity
    data_flat = data.flatten()
    #data_int = np.abs(data_flat)**2

    # Calculate the histogram
    hist, bins = np.histogram(data_flat, bins=50, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Calculate the PDF using Gaussian KDE on the flattened intensity data
    kde = gaussian_kde(data_flat)
    pdf_range = np.linspace(data_flat.min(), data_flat.max(), 300)

    # Plot histogram and PDF
    plt.figure(figsize=(12, 6))
    plt.hist(data_flat, bins=50, density=True, alpha=0.5, color=color, label='Histogram')
    plt.plot(pdf_range, kde(pdf_range), color='black', lw=2, label='PDF')
    plt.title(title)
    plt.xlabel('Intensity')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.savefig(f"{title}.png")
    plt.show()

# Load your image data
sublook1 = np.load('/home/tonix/Documents/Dayana/sublookA.npy')

# Plot for sublook 1
plot_histogram_and_pdf(sublook1, 'Sublook 1', 'green')

# Load sublook2 data and plot, uncomment and load data correctly as needed
sublook2 = np.load('/path_to_sublook2.npy')
plot_histogram_and_pdf(sublook2, 'Sublook 2', 'red')
