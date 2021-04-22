""" A dataset parser that reads images from folders

Folders are scannerd recursively to find image files. Labels are based
on the folder hierarchy, just leaf folders by default.

Hacked together by / Copyright 2020 Ross Wightman
"""
import os

import json

from timm.utils.misc import natural_key

from .parser import Parser
from .class_map import load_class_map
from .constants import IMG_EXTENSIONS


def find_images_and_targets(folder, types=IMG_EXTENSIONS, class_to_idx=None, leaf_name_only=True, sort=True, train_split_percentage=100):
    # Load directly if this information is saved already
    data_root_dir = os.path.dirname(folder)
    # TODO HArdcoded.......fix......
    # data_root_dir = "/cluster/work/cvl/nipopovic/data/ImageNet/2012-1k"
    which_split = os.path.basename(folder)
    if train_split_percentage == 100:
        samples_path = os.path.join(data_root_dir, "partitions", f"{which_split}_samples.json")
    else:
        samples_path = os.path.join(data_root_dir, "partitions", f"{which_split}_samples_{train_split_percentage}.json")
    class_to_idx_path = os.path.join(data_root_dir, "partitions", f"class_to_idx.json")
    if os.path.isfile(samples_path) and os.path.isfile(class_to_idx_path):
        with open(samples_path, "r") as fh:
            images_and_targets = json.load(fh)
            images_and_targets = [(os.path.join(folder, x[0]), x[1]) for x in images_and_targets]
        with open(class_to_idx_path, "r") as fh:
            class_to_idx = json.load(fh)

        return images_and_targets, class_to_idx
    else:
        raise NotImplementedError # TODO The old omplementation can be brought back
    
    # TODO The old implementation
    # labels = []
    # filenames = []
    # for root, subdirs, files in os.walk(folder, topdown=False, followlinks=True):
    #     rel_path = os.path.relpath(root, folder) if (root != folder) else ''
    #     label = os.path.basename(rel_path) if leaf_name_only else rel_path.replace(os.path.sep, '_')
    #     for f in files:
    #         base, ext = os.path.splitext(f)
    #         if ext.lower() in types:
    #             filenames.append(os.path.join(root, f))
    #             labels.append(label)
    # if class_to_idx is None:
    #     # building class index
    #     unique_labels = set(labels)
    #     sorted_labels = list(sorted(unique_labels, key=natural_key))
    #     class_to_idx = {c: idx for idx, c in enumerate(sorted_labels)}
    # images_and_targets = [(f, class_to_idx[l]) for f, l in zip(filenames, labels) if l in class_to_idx]
    # if sort:
    #     images_and_targets = sorted(images_and_targets, key=lambda k: natural_key(k[0]))
    # return images_and_targets, class_to_idx


class ParserImageFolder(Parser):

    def __init__(
            self,
            root,
            class_map='',
            train_split_percentage=100):
        super().__init__()

        self.root = root
        class_to_idx = None
        if class_map:
            class_to_idx = load_class_map(class_map, root)
        self.samples, self.class_to_idx = find_images_and_targets(root, class_to_idx=class_to_idx, train_split_percentage=train_split_percentage)
        if len(self.samples) == 0:
            raise RuntimeError(
                f'Found 0 images in subfolders of {root}. Supported image extensions are {", ".join(IMG_EXTENSIONS)}')

    def __getitem__(self, index):
        path, target = self.samples[index]
        return open(path, 'rb'), target

    def __len__(self):
        return len(self.samples)

    def _filename(self, index, basename=False, absolute=False):
        filename = self.samples[index][0]
        if basename:
            filename = os.path.basename(filename)
        elif not absolute:
            filename = os.path.relpath(filename, self.root)
        return filename
