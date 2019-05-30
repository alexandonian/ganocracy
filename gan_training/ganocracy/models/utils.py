import copy
import numpy as np
import shutil
from scipy.stats import truncnorm

import torch
import torch.nn as nn


class EMA(object):
    """Apply EMA to a model.

    Simple wrapper that applies EMA to a model. Could be better done in 1.0 using
    the parameters() and buffers() module functions, but for now this works
    with state_dicts using .copy_

    """

    def __init__(self, source, target=None, decay=0.9999, start_itr=0):
        self.source = source
        if target is not None:
            self.target = target
        else:
            self.target = copy.deepcopy(source)
        self.decay = decay

        # Optional parameter indicating what iteration to start the decay at.
        self.start_itr = start_itr

        # Initialize target's params to be source's.
        self.source_dict = self.source.state_dict()
        self.target_dict = self.target.state_dict()

        print('Initializing EMA parameters to be source parameters...')
        with torch.no_grad():
            for key in self.source_dict:
                self.target_dict[key].data.copy_(self.source_dict[key].data)

    def update(self, itr=None):
        # If an iteration counter is provided and itr is less than the start itr,
        # peg the ema weights to the underlying weights.
        if itr and itr < self.start_itr:
            decay = 0.0
        else:
            decay = self.decay
        with torch.no_grad():
            for key in self.source_dict:
                self.target_dict[key].data.copy_(self.target_dict[key].data * decay
                                                 + self.source_dict[key].data * (1 - decay))

    def __repr__(self):
        return (f'Source: {type(self.source).__name__}\n'
                f'Target: {type(self.target).__name__}')


def ortho(model, strength=1e-4, blacklist=[]):
    """Apply modified ortho reg to a model.

    This function is an optimized version that directly computes the gradient,
    instead of computing and then differentiating the loss.
    """
    with torch.no_grad():
        for param in model.parameters():
            # Only apply this to parameters with at least 2 axes, and not in the blacklist.
            if len(param.shape) < 2 or any([param is item for item in blacklist]):
                continue
            w = param.view(param.shape[0], -1)
            grad = (2 * torch.mm(torch.mm(w, w.t())
                                 * (1. - torch.eye(w.shape[0], device=w.device)), w))
            param.grad.data += strength * grad.view(param.shape)


def default_ortho(model, strength=1e-4, blacklist=[]):
    """Default ortho regularization.

    This function is an optimized version that directly computes the gradient,
    instead of computing and then differentiating the loss.
    """
    with torch.no_grad():
        for param in model.parameters():
            # Only apply this to parameters with at least 2 axes & not in blacklist.
            if len(param.shape) < 2 or param in blacklist:
                continue
            w = param.view(param.shape[0], -1)
            grad = (2 * torch.mm(torch.mm(w, w.t())
                                 - torch.eye(w.shape[0], device=w.device), w))
            param.grad.data += strength * grad.view(param.shape)


def hook_sizes(G, inputs, verbose=True):

    # Hook the output sizes
    sizes = []

    def hook_output_size(module, input, output):
        sizes.append(output.shape)

    names, mods = zip(*[(name, p) for name, p in G.named_modules() if list(p.parameters()) and (not p._modules)])
    for m in mods:
        m.register_forward_hook(hook_output_size)

    with torch.no_grad():
        output = G(*inputs)

    if verbose:
        max_len = max(max([len(n) for n in names]), len('Input'))

        for i, input in enumerate(inputs):
            print(f'Input {i:<{max_len}} has shape: {input.shape}')

        for name, s in zip(names, sizes):
            print(f'Layer {name:<{max_len}} has shape: {s}')

    return output, names, sizes


def save_checkpoint(state, is_best_IS, is_best_FID, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best_IS:
        fname = filename.rstrip('.pth.tar') + '_best_IS' + '.pth.tar'
        shutil.copyfile(filename, fname)
    if is_best_FID:
        fname = filename.rstrip('.pth.tar') + '_best_FID' + '.pth.tar'
        shutil.copyfile(filename, fname)
