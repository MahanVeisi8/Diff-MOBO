import torch
import numpy as np
import pandas as  pd
import matplotlib.pyplot as plt
from PIL import Image
import  os, sys
from tqdm import tqdm
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")
sys.path.append("..")
sys.path.append("../..")
sys.path.append("src/")
sys.path.append("src/OpenFoam")
sys.path.append("src/diffusion_notebooks")
sys.path.append("data/")
sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("../.."))
from diffusion_core.diffusion import GaussianDiffusion1D
from diffusion_core.model import Unet1D
from pathlib import Path
import scipy.io as sio
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from utils import  BO_surrogate_uncertainty
sys.path.append("../../OpenFoam")
from OpenFoam.Airfoil_simulation_1.ShapeToPerformance import shape_to_performance as STP1
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import yaml

DATA_DIR = Path(rf"../../../data")
coord_mm = np.load(DATA_DIR/"coord_min_max.npy")  # [[x_min,y_min],[x_max,y_max]]
x_min,y_min = coord_mm[0]; x_max,y_max = coord_mm[1]
print(x_min,y_min)

def inv_coords(xs_s):                   # xs_s shape (...,2,192) tensor
    xs_np = xs_s.permute(0,2,1).cpu().numpy()    # -> (B,192,2)
    xs_np[...,0] = xs_np[...,0]*(x_max-x_min) + x_min
    xs_np[...,1] = xs_np[...,1]*(y_max-y_min) + y_min
    return xs_np                                # (B,192,2) numpy

def init_generate_samples_latents(model ,diffusion, checkpoint_path, NUM_TO_GENERATE , BATCH_SIZE):
    """
        input:
            checkpoint_path:    the loading path for the weigths of the
                                diffusion unet.
            NUM_TO_GENERATE
            BATCH_SIZE
        output: 
            all_latents:    list of the latents used for sampling phase
            all_shapes:     list of the shapes  generated from the latents
    """
    # Load checkpoint
    # checkpoint_path = rf"src/diffusion_notebooks/DIffusion_model_weigths_and_datas/dpp_0.1_autonorm_true_125_from_base_ddpm/model_epoch_124.pt"
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.eval()
    print("Loaded model weights from:", checkpoint_path)

    num_to_generate = NUM_TO_GENERATE
    batch_size      = BATCH_SIZE

    all_latent = []
    all_shapes = []
    with torch.no_grad():
        done = 0
        while done < num_to_generate:
            cur = min(batch_size, num_to_generate - done)
            
            latent = torch.randn((cur,2,192)).to(device)
            samples = diffusion.latent_sample(latent , is_ddim=True)
            generated_real = inv_coords(samples)
            
            all_latent.append(latent.cpu().detach().numpy())
            all_shapes.append(generated_real)
            done += cur
            print(f"Generated {done}/{num_to_generate}")

    np.save(os.path.join("Database" , "DB_innerloop.npy") , {
        "latents": np.vstack(all_latent),
        "shapes": np.vstack(all_shapes),
    })


if __name__ == "__main__":

    # reading the yaml file
    with open("innerloop_1_config.yaml", "r") as file:
        config = yaml.safe_load(file)  # Converts YAML → Python dict
    # """
    # The main Hyperparams
    # """
    device = config["model"]['device']
    NUM_TO_GENERATE = config['sampling']["number_to_generate"]
    BATCH_SIZE = config['sampling']["batch_size"]
    Unet_checkpoint_path = config["model"]['unet_checkpoint']

    # Same architecture as in training
    Unet_model = Unet1D(
        dim=32,
        dim_mults=(2, 4, 8, 16),
        channels=2,  # X and Y
        dropout=0.1
    ).to(device)  # or .to(device)

    # Create the same diffusion wrapper
    diffusion = GaussianDiffusion1D(
        Unet_model,
        seq_length=192,      # must match your training setup
        objective='pred_noise',
        timesteps=1000,
        auto_normalize=False
    ).to(device)  # or .to(device)
    init_generate_samples_latents(Unet_model , 
                                    diffusion , 
                                    NUM_TO_GENERATE=NUM_TO_GENERATE,
                                    BATCH_SIZE=BATCH_SIZE,
                                    checkpoint_path=Unet_checkpoint_path
                                    )