import os

from .dataset import IterableImageDataset, ImageDataset, ImageDatasetHDF5


def _search_split(root, split):
    # look for sub-folder with name of split in root and use that if it exists
    split_name = split.split('[')[0]
    try_root = os.path.join(root, split_name)
    try_root = try_root
    if os.path.exists(try_root):
        return try_root
    if split_name == 'validation':
        try_root = os.path.join(root, 'val')
        try_root = try_root
        if os.path.exists(try_root):
            return try_root
    return root
    
def _search_split_hdf5(root, split):
    # look for sub-folder with name of split in root and use that if it exists
    split_name = split.split('[')[0]
    try_root = os.path.join(root, split_name + '.hdf5')
    if os.path.isfile(try_root):
        return try_root
    if split_name == 'validation':
        try_root = os.path.join(root, 'val' + '.hdf5')
        if os.path.isfile(try_root):
            return try_root
    return root


def create_dataset(name, root, split='validation', search_split=True, is_training=False, batch_size=None, **kwargs):
    name = name.lower()
    if name.startswith('tfds'):
        ds = IterableImageDataset(
            root, parser=name, split=split, is_training=is_training, batch_size=batch_size, **kwargs)
    elif name == "hdf5":
        # FIXME support more advance split cfg for ImageFolder/Tar datasets in the future
        kwargs.pop('repeats', 0)  # FIXME currently only Iterable dataset support the repeat multiplier
        if search_split and os.path.isdir(root):
            hdf5_file_path = _search_split_hdf5(root, split)
        ds = ImageDatasetHDF5(hdf5_file_path, parser=name, **kwargs) 
    else:
        # FIXME support more advance split cfg for ImageFolder/Tar datasets in the future
        kwargs.pop('repeats', 0)  # FIXME currently only Iterable dataset support the repeat multiplier
        if search_split and os.path.isdir(root):
            root = _search_split(root, split)
        ds = ImageDataset(root, parser=name, **kwargs)

    print(f"Created {split} data set partition containing {ds.__len__()} data points. (name:{name}, root:{root})\n")
    return ds
