## 1. Configuring and Creating Virtual Machines
    TODO...

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