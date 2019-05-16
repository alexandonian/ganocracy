import torch
import numpy as np
import torchvision.utils as vutils
import matplotlib.pyplot as plt


def visualize_dataset(dataset, num_samples=64, figsize=(15, 15), title='Real Images'):
    data = torch.stack([dataset[i][0] for i in range(num_samples)])
    # Plot the real images
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.title(title)
    plt.imshow(np.transpose(vutils.make_grid(data, padding=5, normalize=True).cpu(), (1, 2, 0)))
