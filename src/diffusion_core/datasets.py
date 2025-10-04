import os
import math
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from accelerate import Accelerator
from ema_pytorch import EMA

from tqdm.auto import tqdm

from denoising_diffusion_pytorch.version import __version__
import numpy as np

# data

class Dataset1D(Dataset):
    def __init__(self, tensor: Tensor):
        super().__init__()
        self.tensor = tensor.clone()

    def __len__(self):
        return len(self.tensor)

    def __getitem__(self, idx):
        return self.tensor[idx].clone()

class AirfoilDataset(Dataset):
    def __init__(self,
                 xs_scaled_path,
                 ys_scaled_path,
                 coord_min_max_path,
                 label_min_max_path):
        self.xs = np.load(xs_scaled_path).astype(np.float32)   # (N,192,2)
        self.ys = np.load(ys_scaled_path).astype(np.float32)   # (N,2)

        coord_mm = np.load(coord_min_max_path)
        label_mm = np.load(label_min_max_path)

        self.x_min,  self.y_min  = coord_mm[0]
        self.x_max,  self.y_max  = coord_mm[1]
        self.cl_min, self.cd_min = label_mm[0]
        self.cl_max, self.cd_max = label_mm[1]

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        coords = torch.from_numpy(self.xs[idx].T)      # (2,192)
        cl, cd = self.ys[idx]
        return coords, torch.tensor([cl, cd], dtype=torch.float32)

    def inverse_scale_coords(self, xs_s):
        """
        xs_s : (..., 192, 2) or (..., 2, 192)  tensor **or** numpy
               If shape is (...,2,192) we treat axis 1 as (x,y) and transpose.
        returns numpy array in physical units, shape (...,192,2)
        """
        # to numpy
        xs_np = xs_s.detach().cpu().numpy() if torch.is_tensor(xs_s) else np.asarray(xs_s)
        # ensure last dim==2
        if xs_np.shape[-2] == 2 and xs_np.shape[-1] == 192:   # (B,2,192)
            xs_np = np.transpose(xs_np, (0,2,1))              # -> (B,192,2)
        xs = xs_np.copy()
        xs[...,0] = xs[...,0]*(self.x_max - self.x_min) + self.x_min
        xs[...,1] = xs[...,1]*(self.y_max - self.y_min) + self.y_min
        return xs                                                 # (B,192,2)

    def inverse_scale_labels(self, ys_s):
        ys_np = ys_s.detach().cpu().numpy() if torch.is_tensor(ys_s) else np.asarray(ys_s)
        cl = ys_np[...,0]*(self.cl_max - self.cl_min) + self.cl_min
        cd = ys_np[...,1]*(self.cd_max - self.cd_min) + self.cd_min
        return np.stack([cl, cd], axis=-1)

