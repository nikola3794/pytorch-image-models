import os
import json
import random

import math

import h5py

import re


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


WHICH_SPLIT = "train"

hdf5_file_path = f"/srv/beegfs02/scratch/hl_task_prediction/data/data_sets/ImageNet/2012-1k/{WHICH_SPLIT}.hdf5"
data_root_dir = os.path.dirname(hdf5_file_path)

# Load class name to index dictionary
class_to_idx_path = os.path.join(data_root_dir, "partitions", f"class_to_idx.json")
if not os.path.isfile(class_to_idx_path):
    # Building class name to index dictionary
    with h5py.File(hdf5_file_path, 'r') as fh:
        unique_labels = set(list(fh))
    sorted_labels = list(sorted(unique_labels, key=natural_key))
    class_to_idx = {c: idx for idx, c in enumerate(sorted_labels)}
    with open(class_to_idx_path, "w") as fh:
        json.dump(class_to_idx, fh)
else:
    # If already saved, load directly
    with open(class_to_idx_path, "r") as fh:
        class_to_idx = json.load(fh)

# Fetch all image relative paths and label names
labels = []
filenames = []
with h5py.File(hdf5_file_path, 'r') as fh:
    for i, cls_dir in enumerate(fh):
        # if i == 3:
        #     break
        print(i)
        img_names = list(fh[cls_dir])
        img_names = [cls_dir + "/" + x for x in img_names]
        
        random.shuffle(img_names)
        random.shuffle(img_names)
        random.shuffle(img_names)

        filenames.append(img_names)
        labels.append(cls_dir)

# Save different partitions 
partitions = [100] if WHICH_SPLIT == "val" else [1, 2, 3, 4, 5, 6, 7, 8, 9] + list(range(10, 101, 5))
for divs in partitions:
    labels_tmp = []
    filenames_tmp = []
    for lbl, fnames in zip (labels, filenames):
        n = round(len(fnames)*(float(divs)/100.0))
        n = 1 if n == 0 else n
        fnm_tmp = fnames[:n]
        filenames_tmp.extend(fnm_tmp)
        labels_tmp.extend([lbl]*n)
        
    images_and_targets = [(f, class_to_idx[l]) for f, l in zip(filenames_tmp, labels_tmp) if l in class_to_idx]
    sort = True
    if sort:
        images_and_targets = sorted(images_and_targets, key=lambda k: natural_key(k[0]))
    
    # Save these lists if they are not saved already for the current dataset
    # (because loading them here takes time)
    if divs != 100:
        samples_path = os.path.join(data_root_dir, "partitions", f"{WHICH_SPLIT}_samples_{divs}.json")
    else:
        samples_path = os.path.join(data_root_dir, "partitions", f"{WHICH_SPLIT}_samples.json")
    if not os.path.isfile(samples_path):
        with open(samples_path, "w") as fh:
            json.dump(images_and_targets, fh)
