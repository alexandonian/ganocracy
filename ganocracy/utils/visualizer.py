import threading

import torch
import numpy as np


import torchvision.utils as vutils
import matplotlib.pyplot as plt


def visualize_data(data, num_samples=64, figsize=(15, 15), title='Real Images'):
    if isinstance(data, torch.utils.data.Dataset):
        print(data)
        samples = torch.stack([data[i][0] for i in range(num_samples)])
    elif isinstance(data, torch.utils.data.DataLoader):
        print(data.dataset)
        samples = next(iter(data))[0][:num_samples]
    else:
        raise ValueError(f'Unrecognized data source type: {type(data)}'
                         'Must be instance of either torch Dataset or DataLoader')
    # Plot the real images
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.title(title)
    plt.imshow(np.transpose(vutils.make_grid(samples, padding=5, normalize=True).cpu(), (1, 2, 0)))


def _save_sample(G, fixed_noise, filename, nrow=8, padding=2, normalize=True):
    fake_image = G(fixed_noise).detach()
    vutils.save_image(fake_image, filename, nrow=nrow, padding=padding, normalize=normalize)


def save_samples(G, fixed_noise, filename, threaded=True):
    if threaded:
        G.to('cpu')
        thread = threading.Thread(name='save_samples',
                                  target=_save_sample,
                                  args=(G, fixed_noise, filename))
        thread.start()
    else:
        _save_sample(G, fixed_noise, filename)


def interp(x0, x1, num_midpoints, device='cuda'):
    """Interpolate between x0 and x1.

    Args:
        x0 (array-like): Starting coord with shape [batch_size, ...]
        x1 (array-like): Ending coord with shape [batch_size, ...]
        num_midpoints (int): Number of midpoints to interpolate.
        device (str, optional): Device to create interp. Defaults to 'cuda'.
    """
    lerp = torch.linspace(0, 1.0, num_midpoints + 2, device=device).to(x0.dtype)
    return torch.lerp(x0, x1, lerp)


num_per_sheet = 4
num_midpoints = 4
dim_z = 1

x0 = torch.randn(num_per_sheet, dim_z)
x1 = torch.randn(num_per_sheet, dim_z)
out = interp(x0, x1, num_midpoints, device='cpu')
