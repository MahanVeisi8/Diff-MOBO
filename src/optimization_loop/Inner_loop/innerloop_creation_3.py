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
sys.path.append(os.path.abspath("../../.."))
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
import scipy.io as sio
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from utils import  *
import yaml
from models import UA_surrogate_model


import argparse

# Create the argument parser
parser = argparse.ArgumentParser(description="Example script with iteration argument")

# Add the --iteration argument
parser.add_argument(
    "--iteration",
    type=int,
    required=True,       # Make it mandatory
    help="Iteration number to run (e.g., --iteration 3)"
)

# Parse the arguments
args = parser.parse_args()
iter = args.iteration

if __name__== "__main__":

    # reading the yaml file
    with open("innerloop_config.yaml", "r") as file:
        config = yaml.safe_load(file)  # Converts YAML → Python dict

    # """
    # The main Hyperparams
    # """

    iterations = config["process"]["iterations"]
    num_cores = config["process"]["number_of_cores"]
    docker_mount_path = config["docker_setup"]["docker_mount_path"]
    docker_container_id = config["docker_setup"]["docker_container_id"]
    NUM_TO_GENERATE = config["sampling"]["number_to_generate"]
    BATCH_SIZE = config["sampling"]["batch_size"]
    # # device = config["model"]["device"]
    device = 'cpu'
    Unet_checkpoint_path = config["model"]["unet_checkpoint"]
    saving_path = rf"src/optimization_loop/Inner_loop/Database/DB_innerloop.npy"
    number_generations = config["genetic_algorithm"]["number_generations"]
    population_size = config["genetic_algorithm"]["population_size"]
    from_DB_innerloop = config["genetic_algorithm"]["from_DB_innerloop"]
    checkpoint_path = config["process"]["checkpoint_path"]

    retrain_epoch = config["UA_surrogate_model"]["epoches"]
    retrain_batch_size = config["UA_surrogate_model"]["batch_size"]
    retrain_learning_rate = config["UA_surrogate_model"]["learning_rate"]
    retrain_patience = config["UA_surrogate_model"]["patience"]
    retrain_LAMBDA_CL = config["UA_surrogate_model"]["LAMBDA_CL"]
    retrain_LAMBDA_CD = config["UA_surrogate_model"]["LAMBDA_CD"]

    
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

    problem_uncertainty = BO_surrogate_uncertainty(diffusion = diffusion,device=device,num_cores=num_cores,n_iter=iter)

    if iter != 0:
        # else loading from the checkpoint and updated weigths (all the weights are there and its  easier to track it)
        problem_uncertainty.UA_surrogate_model.load_state_dict(torch.load(config["UA_surrogate_model"]["saved_update_path"],weights_only=True))
        problem_uncertainty.UA_surrogate_model = problem_uncertainty.UA_surrogate_model.to(config["UA_surrogate_model"]["device"])


    print(f"iteration: {iter}")
    print("Stage 0")
    Retraining_UA_modules(iteration=iter,
                          device=device,
                            num_cores=num_cores,
                            model=problem_uncertainty.UA_surrogate_model,
                            checkpoint_path=checkpoint_path,
                            batch_size=retrain_batch_size,
                            epoches=retrain_epoch,
                            lr=retrain_learning_rate,
                            patience=retrain_patience,
                            LAMBDA_CD=retrain_LAMBDA_CD,
                            LAMBDA_CL=retrain_LAMBDA_CL)


    # Stage 1:
    print("Stage 1")   # Need GPU
    NSGA_BO_surrogate_modules = GEN_UA(config = config,
                                problem_uncertainty=problem_uncertainty,
                                diffusion=diffusion, 
                                device=device,
                                number_iter=iter,
                                num_cores = num_cores, 
                                number_generations=number_generations, 
                                population_size=population_size,
                                from_DB_innerloop=True)

    # Stage 2:
    print("Stage 2")  # Need GPU
    NSGA_latent_to_shape(Unet_model , 
                diffusion , 
                num_cores,
                BATCH_SIZE=BATCH_SIZE,
                checkpoint_path=Unet_checkpoint_path,
                docker_mount_path=docker_mount_path,
                device =  device
                )

