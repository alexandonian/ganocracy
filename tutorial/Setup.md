
## 1. Configuring and Creating Virtual Machines
    TODO...
    System Requirements

### Formatting and mounting a persistent disk

Typically, additional non-boot disks start with no data or file systems. You must format those disks yourself after you attach them to your instances. The formatting process is different on Linux instances and Windows instances. Here, we will focus on preparing a Linux instance.


 1. **Identify the disk device ID:** In the terminal, use the `lsblk` command to list the disks that are attached to your instance and find the disk that you want to format and mount.

    ```
    $ sudo lsblk
    NAME    MAJ:MIN RM  SIZE RO TYPE MOUNTPOINT
    xvda    202:0    0  100G  0 disk
    ├─xvda1 202:1    0  256M  0 part /boot
    └─xvda2 202:2    0 99.8G  0 part /
    xvdb    202:16   0    2G  0 disk
    └─xvdb1 202:17   0    2G  0 part [SWAP]
    xvdc    202:32   0    2T  0 disk  <---- THIS IS THE DISK WE WANT TO USE!
    xvdh    202:112  0   64M  0 disk
    ```
    In this example `xvdc` is the device ID for the disk we want to format and mount.

2. **Format the disk:** You can use any file format that you need, but the most simple method is to format the entire disk with a single `ext4` file system and no partition table.

    Use the `mkfs` tool. **WARNING**: This command deletes all data from the specified disk, so make sure that you specify the disk device correctly. To maximize disk performance, use the recommended formatting options in the `-E` flag. It is not necessary to reserve space for root on this secondary disk, so specify `-m 0` to use all of the available disk space.

    ```
    sudo mkfs.ext4 -m 0 -F -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/[DEVICE_ID]
    ```

    where `[DEVICE_ID]` is the device ID of the disk that you are formatting. For this example, specify `xvdc` to format the entire disk with no partition table.

3. **Create a mount point for the new disk:** You can use any directory that you like, but this example creates a new directory under `/mnt/disks/`.

    ```
    sudo mkdir -p /mnt/disks/[MNT_DIR]
    ```
    where: `[MNT_DIR]` is the directory where you will mount your disk.

4. **Mount Disk**: Use the `mount` tool to mount the disk to the instance with the discard option enabled:
    ```
    sudo mount -o discard,defaults /dev/[DEVICE_ID] /mnt/disks/[MNT_DIR]
    ```
    where:

    `[DEVICE_ID]` is the device ID of the disk that you are mounting.
    `[MNT_DIR]` is the directory where you will mount your disk.

5. **Configure read and write permissions on the device:** For this example, grant write access to the device for all users.

    ```
    sudo chmod a+w /mnt/disks/[MNT_DIR]
    ```
    where: `[MNT_DIR]` is the directory where you mounted your disk.

You should now be able to read an write to this disk!

**TL;DR:** Commands to run:
```
# Mount data drive
DEVICE_ID=xvdc          # UPDATE
MNT_DIR=/mnt/disks/data # UPDATE

sudo mkdir -p $MNT_DIR
sudo mkfs.ext4 -m 0 -F -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/$DEVICE_ID
sudo mount -o discard,defaults /dev/$DEVICE_ID $MNT_DIR
sudo chmod a+w $MNT_DIR
```

### Installing software.

#### The basics
Python is a popular and expressive language that allows rapid prototyping and development, and thus has become a primary workhourse in scientific computing and data science applications, including deep learning. Install one of the latest releases of 64-bit Python 3.  We recommend the Anaconda3 Python distribution, which comes with many commonly used packages, such as numpy, scipy and pandas, already installed.

You can download the Anaconda distribution of python from the [Anaconda website](https://www.anaconda.com/distribution/) or download it directly with the following commands:

```
wget https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
sudo sh Anaconda3-2019.03-Linux-x86_64.sh -b
export PATH=$HOME/anaconda3:$PATH  # Make sure anaconda is added to your path environment variable!
conda update -y conda              # Update conda (-y forces update without prompt)
```

#### Deep learning frameworks

Modern deep learning research relies heavily on powerful frameworks such as [PyTorch](https://pytorch.org/) and [Tensorflow](https://www.tensorflow.org/), which provide high level tools needed to develop and train deep neural networks. Critcally, these frameworks are backed by GPU-accelerated libraries such Nvidia's CUDA, cuDNN and NCCL to deliver high-performance multi-GPU accelerated training.   For a brief overview of popular supported deep learning frameworks, please see Nvidia's [Deep Learning Frameworks](https://developer.nvidia.com/deep-learning-frameworks).

```
conda install -y pytorch==1.1 torchvision cudatoolkit=10.0 -c pytorch
pip install tensorflow-gpu keras visdom gpustat
```


cuda
```
wget https://developer.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda-repo-ubuntu1604-10-1-local-10.1.105-418.39_1.0-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604-10-1-local-10.1.105-418.39_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-<version>/7fa2af80.pub
sudo apt-get update
sudo apt-get install -y --allow-unauthenticated cuda
```

cudnn
```
wget http://visiongpu23.csail.mit.edu/scratch/aandonia/pkg/cudnn-10.1-linux-x64-v7.5.1.10.tgz
tar -xzvf cudnn-10.1-linux-x64-v7.5.1.10.tgz
sudo cp cuda/include/cudnn.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
```

nccl

```
wget http://visiongpu23.csail.mit.edu/scratch/aandonia/pkg/nccl_2.4.2-1+cuda10.1_x86_64.tar.gz
tar -xzf nccl_2.4.2-1+cuda10.1_x86_64.tar.gz
sudo mkdir -p /usr/local/nccl-2.4
sudo cp -vRf nccl_2.4.2-1+cuda10.1_x86_64/* /usr/local/nccl-2.4
sudo ln -s /usr/local/nccl-2.4/include/nccl.h /usr/include/nccl.h
```