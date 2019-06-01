import threading

import torch
import numpy as np
from scipy.stats import truncnorm


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
    visualize_samples(samples, figsize=figsize, title=title)


def visualize_samples(samples, figsize=(15, 15), title='Samples',
                      nrow=8, padding=5, normalize=True):
    # Plot the real images
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.title(title)
    im = vutils.make_grid(samples, nrow=nrow, padding=padding, normalize=normalize).cpu()
    plt.imshow(np.transpose(im, (1, 2, 0)))


def smooth_data(data, amount=1.0):
    if not amount > 0.0:
        return data
    data_len = len(data)
    ksize = int(amount * (data_len // 2))
    kernel = np.ones(ksize) / ksize
    return np.convolve(data, kernel, mode='same')


def plot_loss_logs(G_loss, D_loss, figsize=(15, 5), smoothing=0.001):
    G_loss = smooth_data(G_loss, amount=smoothing)
    D_loss = smooth_data(D_loss, amount=smoothing)
    plt.figure(figsize=figsize)
    plt.plot(D_loss, label='D_loss')
    plt.plot(G_loss, label='G_loss')
    plt.legend(loc='lower right', fontsize='medium')
    plt.xlabel('Iteration', fontsize='x-large')
    plt.ylabel('Losses', fontsize='x-large')
    plt.title('Training History', fontsize='xx-large')
    plt.show()

    
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
    x0 = x0.view(x0.size(0), 1, *x0.shape[1:])
    x1 = x1.view(x1.size(0), 1, *x1.shape[1:])
    lerp = torch.linspace(0, 1.0, num_midpoints + 2, device=device).to(x0.dtype)
    lerp = lerp.view(1, -1, 1)
    return torch.lerp(x0, x1, lerp)


def truncated_z_sample(batch_size, dim_z, truncation=1.0, seed=None, device='cuda'):
    state = None if seed is None else np.random.RandomState(seed)
    values = truncnorm.rvs(-2, 2, size=(batch_size, dim_z), random_state=state)
    return torch.Tensor(float(truncation) * values).to(device)
