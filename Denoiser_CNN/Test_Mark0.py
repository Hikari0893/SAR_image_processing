import matplotlib.pyplot as plt
from Mark_0 import *

checkpoint = '/home/tonix/Documents/Dayana/NET/tb_logs/AC_Net/version_0/checkpoints/epoch=9-step=28220.ckpt'
# Instantiate the Model
model = Autoencoder_ACDC.load_from_checkpoint(checkpoint)

# DataLoader for test data
test_loader = model.predict_dataloader()
# Get a batch of data
images = next(iter(test_loader))
# Make sure the model is in evaluation mode and is not calculating gradients
model.eval()

with torch.no_grad():
    # Get the reconstructions
    reconstructions = model.predict_step(images,0)

# View data in groups of num_images images
num_images = 1  # Number of images to display
for i in range(num_images):
    # View original image
    plt.subplot(2, num_images, i + 1)
    plt.imshow(images[i+10].squeeze(), cmap='gray')
    plt.axis('off')
    plt.title(f"Original {i+1}")

    # Display reconstructed images
    plt.subplot(2, num_images, num_images + i + 1)
    plt.imshow(reconstructions[i+10].squeeze(), cmap='gray')
    plt.axis('off')
    plt.title(f"Reconstruida {i+1}")

plt.show()