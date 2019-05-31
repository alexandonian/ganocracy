# GAN Training Tutorial Setup


This repo has been tested with Python 3.6+ and Pytorch 1.0+. **Note:** While it may be possible to make use of this repo with earlier versions of Python and PyTorch, it has not been tested and will likely require small modifications scattered throughout the code.

### Note:

If you have already completed the configuration for the [GANdissect notebook](../gandissect), you may be able to use the `netd` environment for this notebook as well without any further configuration:

```
conda activate netd
```

### Requirements:
- Python 3.6 or greater.
- PyTorch 1.1 (LTS). Detailed installation instructions can be found [here](https://pytorch.org/get-started/locally/).
- torchvision 0.3.0. **Note**: Version 0.3.0 was just announced on May 23, 2019, so existing installations may need updating.
- tqdm, numpy, scipy, and h5py
- moviepy (optional)

If your system does not already meet these requirements, we recommend downloading the Anaconda Distribution of Python 3. Anaconda comes with the package manager `conda` (installation instructions can be found [here](http://ganocracy.csail.mit.edu/tutorial/setup.html)),  which makes installing PyTorch and other dependencies much easier.

After installing `conda`, you can create an environment for this