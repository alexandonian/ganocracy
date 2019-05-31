# GAN Training Tutorial Setup


This repo has been tested with Python 3.6+ and Pytorch 1.0+. **Note:** While it may be possible to make use of this repo with earlier versions of Python and PyTorch, it has not been tested and will likely require small modifications scattered throughout the code.

### Note:

If you have already completed the configuration for the [GANdissect notebook](../gandissect), you may be able to use the `netd` environment for this notebook as well without any further configuration:

```
conda activate netd
```

Otherwise, continue reading for instructions on how to get set up to run this notebook.

### Requirements:
- Python 3.6 or greater.
- PyTorch 1.1 (LTS). Detailed installation instructions can be found [here](https://pytorch.org/get-started/locally/).
- torchvision 0.3.0. **Note**: Version 0.3.0 was just announced on May 23, 2019, so existing installations may need updating.
- tqdm, numpy, scipy, and h5py
- moviepy (optional)

**Step 1: Download Anaconda** If your system does not already meet these requirements, we recommend downloading the Anaconda Distribution of Python 3. Anaconda comes with the package manager `conda` (installation instructions can be found [here](http://ganocracy.csail.mit.edu/tutorial/setup.html)),  which makes installing PyTorch and other dependencies much easier.

**Step 2 (optional): Create `conda` environment** After installing `conda`, you can create an environment for this project to manage dependencies:

```
# Create a conda environment called 'gantraining'
conda create --name gantraining

# To activate this environment, use:
conda activate gantraining
```

**Step 3: Install PyTorch and dependencies:**
	
Installing the latest version of PyTorch and torchvision can be done in a single line:

```	
conda install pytorch torchvision -c pytorch
```
and the dependencies with:
	
```
conda install h5py
pip install tqdm moviepy
```

### Starting the Jupyter Notebook
	
**Locally:** You can start a jupyter notebook from this directory (`gan_training`) with the following:
	
```
jupyter notebook
```
A browser window shoud automatically open.
	
**Remotely with SSH Port-Forwarding:** If you are fortunate to have access to a headless remote server, ideally with several GPUs, it is possible to run the notebook on the server and still view it locally on your personal machine via ssh port-forwarding. Feel free to read more about ssh port-forwarding on your own, but it's not necessary for this tutorial.
 
 Using port forwarding to view juypter notebooks on a remote server requires two steps:
 
1. **On the REMOTE machine**: Start the jupyter notebook as normal. You may need to include the `--no-browser` option:
    ```
    cd tutorial
    jupyter notebook --no-browser
    ```
    Jupyter notebook will display a bunch of information, which should include something like:
    ```
    [I 18:20:18.770 NotebookApp] The Jupyter Notebook is running at: http://localhost:8888/?token=MY_TOKEN
    [I 18:20:18.770 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
    [C 18:20:18.771 NotebookApp]
    
        Copy/paste this URL into your browser when you connect for the first time,
        to login with a token:
            http://localhost:8888/?token=MY_TOKEN
    ```
    
    Make note of the port number and token. In this case, we have `REMOTE_PORT=8888`

2. **On the LOCAL machine:** Decide which port you want to use to view the notebook. Choose anything in the range 1024-49151  (unreserved "user" ports). Call the port forwarding command:

    ```
    LOCAL_PORT=8888
    REMOTE_PORT=8888  # From step 1.
    
    ssh -fNL $LOCAL_PORT:localhost:$REMOTE_PORT $REMOTE_IP_ADDR
    ```
    In your browser, navigate to `localhost:$LOCAL_PORT`. If you are prompted, enter your token from step 1. You should now be able to see your jupyter notebooks running on the remote server!