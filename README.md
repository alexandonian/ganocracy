# GANocracy: Democratizing GANs



This repository contains the two Jupyter notebooks used during the May 31, 2019 [GANocracy tutorial](http://ganocracy.csail.mit.edu/tutorial/tutorial.html) at MIT.

#### Setup
Please complete the [setup instructions](http://ganocracy.csail.mit.edu/tutorial/setup.html) before running the notebooks.

### 1. [Exploring a Generator with GANdissect](gandissect)
by: [David Bau](https://people.csail.mit.edu/davidbau/home/), MIT

When GANs generate images, are they simply reproducing memorized pixel patterns, or are they composing images from learned objects? How do different architectures affect what the GAN learns? Which neurons are responsible for undesirable artifacts in generated images?

GANdissect [[GitHub](https://github.com/CSAILVision/gandissect), [paper](https://arxiv.org/pdf/1811.10597.pdf)] is an analytic framework for visualizing the internal representations of a GAN generator at the unit-, object-, and scene-level.

<img src="gandissect/assets/GANdissect.jpg" width="700" height="700">
<sup>Image credit: Bau, David, et al. ["GAN Dissection: Visualizing and Understanding Generative Adversarial Networks."](https://arxiv.org/pdf/1811.10597.pdf) arXiv preprint arXiv:1811.10597 (2018).</sup>

GANdissect helps shed light on what representations GANs are learning across layers, models, and datasets, and we can use that knowledge to compare, improve, and better control GAN performance.

### 2. [Training a GAN](gan_training)
by: [Alex Andonian](https://www.alexandonian.com/), MIT

How do you actually build and train a GAN? What are best practices, tips, and tricks to help simplify the process? 

This notebook offers a step-by-step walk-through in PyTorch of Deep Convolutional (DCGAN) and Conditional (cGAN) GAN training, from data preparation and ingestion through results analysis.

<img src="gan_training/assets/dcgan_progress.gif" width="700" height="700">
<sup>Image credit: Alex Andonian</sup>
