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
