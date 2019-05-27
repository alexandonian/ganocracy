# Ganocracy: Democratizing GANs

## Introduction:
[TODO]
## For Tutorial Attendees.
[TODO]
#### Before you arrive
[TODO]
## Installation

This repo has been tested with Python 3.6+ and Pytorch 1.0+. **Note:** While it may be possible to make use of this repo with earlier versions of Python and PyTorch, it has not been tested and will likely require small modifications scattered throughout the code.

### Requirements:
- Python 3.6 or greater.
- PyTorch 1.1 (LTS). Detailed installation instructions can be found in [here](https://pytorch.org/get-started/locally/).
- torchvision 0.3.0. **Note**: Version 0.3.0 was just announced on May 23, 2019, so existing installations may need updating.
- tqdm, numpy, scipy, and h5py
- moviepy (optional)

If your system does not already meet these requirements, we recommend downloading the Anaconda Distribution of Python 3. Anaconda comes with the package manager `conda`,  which makes installing PyTorch and other dependencies much easier. See below for step-by-step instructions.

### Step-by-step Instructions:
#### macOS
**Step 1: Install Anaconda.** While many systems come with Python already installed, it may not be the latest version. For example, macOS comes with Python 2.7 by default, but this is not supported by the tutorial.

```sh
# Download installation scripts from Anaconda website:
curl -O https://repo.anaconda.com/archive/Anaconda3-2019.03-MacOSX-x86_64.sh
# Run the interactive installation script:
sh Anaconda3-2019.03-MacOSX-x86_64.sh # and follow the prompts. The defaults are generally good.
```

**Verify your installation:** If you accepted the defaults above, your system should now be using anaconda python by default. To verify that everything is setup properly, check that `which conda`, `which pip` and `which python` all point to the correct path. Additionally, you can check you installed the right version of python with `python --version`.

**Step 2 (optional): Create Conda Environment.** Each project typically has its own set of requirments for specific python packages and versions. When working on multiple projects simultaneously, it can cumbersome and difficult to manage which packages and versions are being used for any given project. In response, package managers like `conda` allow you to create isolated environments as a way to avoid dependency conflicts.

```sh
# Create a conda environment called 'ganocracy'
conda create --name ganocracy
# To activate this environment, use:
conda activate ganocracy
```

**Step 3: Intall PyTorch and Dependencies:**

Installing that latest version of PyTorch and torchvision can be done in single line:

```sh
conda install pytorch torchvision -c pytorch
```

and the dependencies with:

```sh
conda install h5py
pip install tqdm moviepy
```

#### Starting a Jupyter Notebook

**Locally:** You can navigate to the directory of interest `tutorial` and start a jupyter notebook with the following:

```sh
cd tutorial
jupyter notebook
```
A browser window shoud automatically open.

**Remotely with SSH Port-Forwarding:** If you are fortunate to have access to a headless remote server, ideally with several GPUs, it is possible to run the notebook on the server and still view it locally on your personal machine via ssh port-forwarding. Feel free to read more about ssh port-forwarding on your own, but it's not necesary for this tutorial. Using port forwarding to view juypter notebooks on a remote server requires two steps:
1. **On the REMOTE machine**: Start the jupyter notebook as normal. You may need to include the `--no-browser` option:
    ```sh
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

Make note of the port number and token. In this case, we have `REMOTE_PORT=8888` and

2. **On the LOCAL machine:** Decide which port you want to use to view the notebook. Choose anything in the range 1024-49151  (unreserved "user" ports). Call the port forwarding command:

```
LOCAL_PORT=8888
REMOTE_PORT=8888  # From step 1.

ssh -fNL $LOCAL_PORT:localhost:$REMOTE_PORT $REMOTE_IP_ADDR
```
In your browser, navigate to `localhost:$LOCAL_PORT`. If you are prompted, enter your token from step 1. You should now be able to see your jupyter notebooks running on the remote server!
