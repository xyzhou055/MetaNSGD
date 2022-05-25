import os
import re
import copy
from typing_extensions import ParamSpec
import torch
import numpy as np


# Those two functions are taken from torchvision code because they are not available on pip as of 0.2.0
def list_dir(root, prefix=False):
    """List all directories at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the directories found
    """
    root = os.path.expanduser(root)
    directories = list(
        [p for p in os.listdir(root) if os.path.isdir(os.path.join(root, p))]
    )

    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]

    return directories


def list_files(root, suffix, prefix=False):
    """List all files ending with a suffix at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            It uses the Python "str.endswith" method and is passed directly
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the files found
    """
    root = os.path.expanduser(root)
    files = list(
        [p for p in os.listdir(root) if os.path.isfile(os.path.join(root, p)) and p.endswith(suffix)]
    )

    if prefix is True:
        files = [os.path.join(root, d) for d in files]

    return files

def var_add(var_sum, paras):
    if not var_sum:
        return [para*2*0.5 for para in paras]
    else:
        return [v + val for v, val in zip(var_sum, paras)]


def var_substract(paras_1, paras_2):
    return [v - val for v, val in zip(paras_1, paras_2)]

def var_scale(var_sum, scale):
    return [v*scale for v in var_sum]


def gradient_clipping(gradient, L):
    norm = 0
    for para in gradient:
        norm += torch.sum(para ** 2).item()
    norm = np.sqrt(norm)
    if norm > L:
        gradient = [g *(L/norm) for g in gradient]
    return gradient

def find_latest_file(folder):
    files = []
    for fname in os.listdir(folder):
        s = re.findall(r'\d+', fname)
        if len(s) == 1:
            files.append((int(s[0]), fname))
    if files:
        return max(files)[1]
    else:
        return None

pass