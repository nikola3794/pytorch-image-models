""" A dataset parser that reads images from folders

Folders are scannerd recursively to find image files. Labels are based
on the folder hierarchy, just leaf folders by default.

Hacked together by / Copyright 2020 Ross Wightman
"""
import os

from timm.utils.misc import natural_key

from .parser import Parser
from .class_map import load_class_map
from .constants import IMG_EXTENSIONS

import json

import h5py


def find_images_and_targets_in_hdf5(hdf5_path, types=IMG_EXTENSIONS, class_to_idx=None, leaf_name_only=True, sort=True):
    # Load directly if this information is saved already
    data_root_dir = os.path.dirname(hdf5_path)
    which_split = os.path.basename(hdf5_path).split(".")[0]
    samples_path = os.path.join(data_root_dir, f"{which_split}_samples.json")
    class_to_idx_path = os.path.join(data_root_dir, f"{which_split}_class_to_idx.json")
    if os.path.isfile(samples_path) and os.path.isfile(class_to_idx_path):
        with open(samples_path, "r") as fh:
            images_and_targets = json.load(fh)
            images_and_targets = [tuple(x) for x in images_and_targets]
        with open(class_to_idx_path, "r") as fh:
            class_to_idx = json.load(fh)
        return images_and_targets, class_to_idx

    labels = []
    filenames = []
    with h5py.File(hdf5_path, 'r') as fh:
        for i, cls_dir in enumerate(fh):
            # Option 1
            img_names = list(fh[cls_dir])
            img_names = [cls_dir + "/" + x for x in img_names]
            filenames.extend(img_names)
            labels.extend([cls_dir] * len(img_names))

            # Option 2 
            # for img_name in fh[cls_dir]:
            #     base, ext = os.path.splitext(img_name)
            #     if ext.lower() in types:
            #         filenames.append(cls_dir + "/" + img_name)
            #         labels.append(cls_dir)
    if class_to_idx is None:
        # building class index
        unique_labels = set(labels)
        sorted_labels = list(sorted(unique_labels, key=natural_key))
        class_to_idx = {c: idx for idx, c in enumerate(sorted_labels)}
    images_and_targets = [(f, class_to_idx[l]) for f, l in zip(filenames, labels) if l in class_to_idx]
    if sort:
        images_and_targets = sorted(images_and_targets, key=lambda k: natural_key(k[0]))
    
    # Save these lists if they are not saved already for the current dataset
    # (because loading them here takes time)
    data_root_dir = os.path.dirname(hdf5_path)
    which_split = os.path.basename(hdf5_path).split(".")[0]
    samples_path = os.path.join(data_root_dir, f"{which_split}_samples.json")
    if not os.path.isfile(samples_path):
        with open(samples_path, "w") as fh:
            json.dump(images_and_targets, fh)
    class_to_idx_path = os.path.join(data_root_dir, f"{which_split}_class_to_idx.json")
    if not os.path.isfile(class_to_idx_path):
        with open(class_to_idx_path, "w") as fh:
            json.dump(class_to_idx, fh)
    return images_and_targets, class_to_idx
    

class ParserImageInHDF5(Parser):

    def __init__(
            self,
            hdf5_path,
            class_map=''):
        super().__init__()

        self.hdf5_path = hdf5_path
        class_to_idx = None
        if class_map:
            class_to_idx = load_class_map(class_map, hdf5_path)
        self.samples, self.class_to_idx = find_images_and_targets_in_hdf5(hdf5_path, class_to_idx=class_to_idx)
        if len(self.samples) == 0:
            raise RuntimeError(
                f'Found 0 images in subfolders of {hdf5_path}. Supported image extensions are {", ".join(IMG_EXTENSIONS)}')

    def __getitem__(self, index):
        hdf5_rel_path, target = self.samples[index]
        return hdf5_rel_path, target

    def __len__(self):
        return len(self.samples)

    def _filename(self, index, basename=False, absolute=False):
        filename = self.samples[index][0]
        if basename:
            filename = os.path.basename(filename)
        elif not absolute:
            filename = os.path.relpath(filename, self.root)
        return filename
