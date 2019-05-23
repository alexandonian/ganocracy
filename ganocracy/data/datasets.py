import errno
import os
import re
import sys
import tarfile
import warnings

import numpy as np
import h5py as h5
from PIL import Image

import torch
import torch.hub
import torch.utils.data as data
from tqdm import tqdm

from . import transforms

try:
    getattr(torch.hub, 'HASH_REGEX')
except AttributeError:
    torch.hub.HASH_REGEX = re.compile(r'-([a-f0-9]*)\.')


ROOT_URL = 'http://ganocracy.csail.mit.edu/data/'
data_urls = {
    'celeba': {
        'tar': os.path.join(ROOT_URL, 'celeba-054b22a6.tar.gz')
    },
    'buildings_hq': {
        'tar': os.path.join(ROOT_URL, 'buildings_hq.tar.gz'),
        'hdf5': {
            '128': os.path.join(ROOT_URL, 'B128.hdf5'),
            '256': os.path.join(ROOT_URL, 'B256.hdf5'),
        },
    },
    'satellite_images': {
        'tar': os.path.join(ROOT_URL, 'satellite_images-79716c2f.tar.gz')
    },
    'imagenet': {
        'tar': os.path.join(ROOT_URL, 'imagenet.tar.gz'),
        'hdf5': {
            '64': os.path.join(ROOT_URL, 'I64.hdf5'),
            '128': os.path.join(ROOT_URL, 'I128.hdf5'),
            '256': os.path.join(ROOT_URL, 'I256.hdf5'),
        },
    },
    'places365': {
        'tar': os.path.join(ROOT_URL, 'places365.tar.gz'),
        'hdf5': {
            '64': os.path.join(ROOT_URL, 'P64.hdf5'),
            '128': os.path.join(ROOT_URL, 'P128.hdf5'),
            '256': os.path.join(ROOT_URL, 'P256.hdf5'),
        }
    }
}


def load_data_from_url(url, root_dir=None, progress=True):
    cached_file = _load_file_from_url(url, root_dir=root_dir, progress=progress)
    # match = torch.hub.HASH_REGEX.search(cached_file)
    # data_dir = cached_file[:match.start()]

    with tarfile.open(cached_file) as tf:
        name = tf.getnames()[0]
    data_dir = os.path.join(root_dir, name)

    if not os.path.exists(data_dir):
        print(f'Extracting:  "{cached_file}" to {data_dir}')
        with tarfile.open(name=cached_file) as tar:
            # Go over each member
            for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers())):
                # Extract member
                tar.extract(member=member, path=root_dir)

        # tf = tarfile.open(cached_file)
        # print(f'Extracting to: {data_dir}')
        # tf.extractall(path=root_dir)
        # print(f'finished extracting to: {root_dir}')
    else:
        print(f'Data found at: {data_dir}')
    return data_dir


def _load_file_from_url(url, root_dir=None, progress=True):
    r"""Loads the dataset file from the given URL.

    If the object is already present in `root_dir`, it's deserialized and
    returned. The filename part of the URL should follow the naming convention
    ``filename-<sha256>.ext`` where ``<sha256>`` is the first eight or more
    digits of the SHA256 hash of the contents of the file. The hash is used to
    ensure unique names and to verify the contents of the file.

    The default value of `model_dir` is ``$TORCH_HOME/checkpoints`` where
    environment variable ``$TORCH_HOME`` defaults to ``$XDG_CACHE_HOME/torch``.
    ``$XDG_CACHE_HOME`` follows the X Design Group specification of the Linux
    filesytem layout, with a default value ``~/.cache`` if not set.

    Args:
        url (string): URL of the object to download
        data_dir (string, optional): directory in which to save the object
        progress (bool, optional): whether or not to display a progress bar to stderr

    # 'https://pytorch.org/docs/stable/_modules/torch/hub.html#load'

    """
    # Issue warning to move data if old env is set
    if os.getenv('TORCH_MODEL_ZOO'):
        warnings.warn('TORCH_MODEL_ZOO is deprecated, please use env TORCH_HOME instead')

    if root_dir is None:
        torch_home = torch.hub._get_torch_home()
        root_dir = os.path.join(torch_home, 'data')

    try:
        os.makedirs(root_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    parts = torch.hub.urlparse(url)
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(root_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = torch.hub.HASH_REGEX.search(filename).group(1)
        torch.hub._download_url_to_file(url, cached_file, hash_prefix, progress)
    return cached_file


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(dir):
    classes = sorted([d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))])
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in tqdm(sorted(os.listdir(dir))):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

    for root, _, fnames in sorted(os.walk(d)):
        for fname in tqdm(sorted(fnames)):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                item = (path, class_to_idx[target])
                images.append(item)
    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolder(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

        root/dogball/xxx.png
        root/dogball/xxy.png
        root/dogball/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, load_in_mem=False,
                 index_filename=None, **kwargs):
        classes, class_to_idx = find_classes(root)

        # Load pre-computed image directory walk
        if index_filename is None:
            index_filename = os.path.join(root, os.path.basename(root) + '.npz')

        if os.path.exists(index_filename):
            print('Loading pre-saved index file {}'.format(index_filename))
            imgs = np.load(index_filename)['imgs']
        # If first time, walk the folder directory and save the
        # results to a pre-computed file.
        else:
            print('Generating index file {}'.format(index_filename))
            imgs = make_dataset(root, class_to_idx)
            np.savez_compressed(index_filename, **{'imgs': imgs})
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.load_in_mem = load_in_mem

        if self.load_in_mem:
            print('Loading all images into memory...')
            self.data, self.labels = [], []
            for index in tqdm(range(len(self.imgs))):
                path, target = self.transform(imgs[index][0]), imgs[index][1]
                self.data.append(self.loader(path))
                self.labels.append(target)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        if self.load_in_mem:
            img = self.data[index]
            target = self.labels[index]
        else:
            path, target = self.imgs[index]
            try:
                img = self.loader(str(path))
            except OSError:
                return self.__getitem__(min(index + 1, len(self)))
            if self.transform is not None:
                img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, int(target)

    def __len__(self):
        return len(self.imgs)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of classes: {}\n'.format(len(self.classes))
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class SingleImageFolder(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, load_in_mem=False,
                 index_filename='imagenet_imgs.npz', **kwargs):
        # Load pre-computed image directory walk
        if os.path.exists(index_filename):
            print('Loading pre-saved index file {}'.format(index_filename))
            imgs = np.load(index_filename)['imgs']
        # If first time, walk the folder directory and save the
        # results to a pre-computed file.
        else:
            print('Generating index file {}'.format(index_filename))
            imgs = []
            fnames = os.listdir(root)
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, 0)
                    imgs.append(item)
            np.savez_compressed(index_filename, **{'imgs': imgs})
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.load_in_mem = load_in_mem

        if self.load_in_mem:
            print('Loading all {} images into memory...'.format(self.root))
            self.data, self.labels = [], []
            for index in tqdm(range(len(self.imgs))):
                path, target = self.transform(imgs[index][0]), imgs[index][1]
                self.data.append(self.loader(path))
                self.labels.append(target)


def hdf5_transform(img):
    return ((torch.from_numpy(img).float() / 255) - 0.5) * 2


class ImageHDF5(data.Dataset):

    def __init__(self, root, transform=hdf5_transform, target_transform=None,
                 load_in_mem=False, train=True, download=False, validate_seed=0,
                 val_split=0, **kwargs):  # last four are dummies

        self.root = root
        self.target_transform = target_transform
        with h5.File(root, 'r') as f:
            self.num_imgs = len(f['labels'])
            self.num_classes = len(np.unique(f['labels']))

        # Set the transform here.
        self.transform = transform

        # Load the entire dataset into memory?
        self.load_in_mem = load_in_mem

        # If loading into memory, do so now.
        if self.load_in_mem:
            print('Loading {} into memory...'.format(root))
            with h5.File(root, 'r') as f:
                self.data = f['imgs'][:]
                self.labels = f['labels'][:]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        # If loaded the entire dataset in RAM, get image from memory
        if self.load_in_mem:
            img = self.data[index]
            target = self.labels[index]

        # Else load it from disk
        else:
            with h5.File(self.root, 'r') as f:
                img = f['imgs'][index]
                target = f['labels'][index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, int(target)

    def __len__(self):
        return self.num_imgs

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of classes: {}\n'.format(self.num_classes)
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def make_hdf5(dataloader, root, filename, chunk_size=500, compression=False):
    path = os.path.join(root, filename)
    if not os.path.exists(path):
        _make_hdf5(dataloader, root, filename,
                   chunk_size=chunk_size,
                   compression=compression)
    else:
        print('HDF5 file {} already exists!'.format(path))
    return path


def _make_hdf5(dataloader, root, filename, chunk_size=500, compression=False):
    # HDF5 supports chunking and compression. You may want to experiment
    # with different chunk sizes to see how it runs on your machines.
    # Chunk Size/compression     Read speed @ 256x256   Read speed @ 128x128  Filesize @ 128x128    Time to write @128x128
    # 1 / None                   20/s
    # 500 / None                 ramps up to 77/s       102/s                 61GB                  23min
    # 500 / LZF                                         8/s                   56GB                  23min
    # 1000 / None                78/s
    # 5000 / None                81/s
    # auto:(125,1,16,32) / None                         11/s                  61GB

    print('Starting to load {} into an HDF5 file with chunk size {} and compression {}...'.format(filename, chunk_size, compression))

    # Loop over train loader
    dataset_len = len(dataloader.dataset)
    for i, (x, y) in enumerate(tqdm(dataloader)):
        # Stick x into the range [0, 255] since it's coming from the train loader
        x = (255 * ((x + 1) / 2.0)).byte().numpy()
        # Numpyify y
        y = y.numpy()

        if i == 0:  # If we're on the first batch, prepare the hdf5.
            with h5.File(os.path.join(root, filename), 'w') as f:
                print('Producing dataset of len {}'.format(dataset_len))
                maxshape = (dataset_len, x.shape[-3], x.shape[-2], x.shape[-1])
                chunks = (chunk_size, x.shape[-3], x.shape[-2], x.shape[-1])
                imgs_dset = f.create_dataset('imgs', x.shape, dtype='uint8',
                                             maxshape=maxshape,
                                             chunks=chunks,
                                             compression=compression)
                print('Image chunks chosen as {}'.format(imgs_dset.chunks))
                imgs_dset[...] = x

                labels_dset = f.create_dataset('labels', y.shape, dtype='int64',
                                               maxshape=(dataset_len,),
                                               chunks=(chunk_size,),
                                               compression=compression)
                print('Label chunks chosen as {}'.format(labels_dset.chunks))
                labels_dset[...] = y

        else:  # Append to the hdf5.
            with h5.File(os.path.join(root, filename), 'a') as f:
                f['imgs'].resize(f['imgs'].shape[0] + x.shape[0], axis=0)
                f['imgs'][-x.shape[0]:] = x
                f['labels'].resize(f['labels'].shape[0] + y.shape[0], axis=0)
                f['labels'][-y.shape[0]:] = y


def get_dataset(name, root_dir=None, resolution=128, filetype='tar'):
    if filetype == 'tar':
        url = data_urls[name]['tar']
        data_dir = load_data_from_url(url, root_dir)
        dataset = ImageFolder(root=data_dir,
                              transform=transforms.Compose([
                                  transforms.CenterCropLongEdge(),
                                  transforms.Resize(resolution),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5),
                                                       (0.5, 0.5, 0.5))
                              ]))
    elif filetype == 'hdf5':
        url = data_urls[name]['hdf5'][resolution]
        hdf5_file = load_data_from_url(url, root_dir)
        dataset = ImageHDF5(hdf5_file)
    else:
        raise ValueError('Unreconized filetype: {}'.format(filetype))

    return dataset


def old_get_dataset(name, hdf5=True, size=64, targets=False):
    pass


def imagenet(data_dir, size=64, targets=False):
    pass


def places365(data_dir, size=64, targets=False):
    pass


def ffhq(data_dir, size=64, targets=False):
    pass
