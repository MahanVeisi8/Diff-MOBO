import numpy as np
import os , sys
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
import torch
import os,sys
import torch
import os
from pathlib import Path
from OpenFoam.Airfoil_simulation_1.ShapeToPerformance import shape_to_performance as STP1
import pickle

# docker_mount_path = "/home/bardiya/projects/diffusion_air_manifolding/codes/creative-generativeai-diffusion/src/OpenFoam"
# docker_mount_path = "/home/bardiya/projects/diffusion_air_manifolding/codes/Airfoil_MPI_system"
docker_mount_path = "src/OpenFoam"
BATCH_SIZE = 256
# NUM_TO_GENERATE = 2
# BATCH_SIZE = 2
docker_container_id = "e14b8f8d728c" 
saving_path = rf"src/optimization_loop/run_results_xstest.npy"
# device = "cpu"
device = "cuda"

"""
******************************
******************************
warning:
    before running make sure to copy src/OpenFoam/Airfoil_simulation_1/OpenFOAM_0
    in that folder 200 times with new directories name as src/OpenFoam/Airfoil_simulation_1/OpenFOAM_i for the i'th core ussage
******************************
******************************
"""



DATA_DIR = Path(rf"data")
coord_mm = np.load(DATA_DIR/"coord_min_max.npy")  # [[x_min,y_min],[x_max,y_max]]
x_min,y_min = coord_mm[0]; x_max,y_max = coord_mm[1]

def inv_coords(xs_s):                   # xs_s shape (...,2,192) tensor
    xs_np = xs_s.permute(0,2,1).cpu().numpy()    # -> (B,192,2)
    xs_np[...,0] = xs_np[...,0]*(x_max-x_min) + x_min
    xs_np[...,1] = xs_np[...,1]*(y_max-y_min) + y_min
    return xs_np                                # (B,192,2) numpy



if __name__ == "__main__":

    print("loading DB...")
    db2_path = os.path.join(docker_mount_path , "xs_test.npy")
    all_shapes = np.load(db2_path)
    all_latent = np.zeros((all_shapes.shape[0],2,192))  # dummy latents

    data = {
        "latents": all_latent,
        "shapes": all_shapes,
        "performances": None
    }
    
    with open(os.path.join(docker_mount_path , "DB2_xstest.npy"), "wb") as f:
        pickle.dump(data, f, protocol=4) # Compatible with python 3.6.9 in the docker


    print("================================")
    print(rf"starting openfoam in mount path {docker_mount_path} with docker container {docker_container_id}")
    command = (
        f'docker exec {docker_container_id} bash -c "'
        f'source /opt/openfoam5/etc/bashrc && '
        f'cd /home/airfoil_UANA && '
        f'python3 performance_finding_b.py"'
    )

    os.system(command)
    print("================================")

    performance_path =os.path.join(docker_mount_path , rf"performance.npy")
    perfromance = np.load(performance_path,allow_pickle=True)

    # Performance is CL, CD, Index
    # Results is a dict of (latent, shape, performance)
    # Performance should sorted by index to match shapes and latents

    # sort performance based on index, ascending
    perfromance = sorted(perfromance, key=lambda x: x[2])  # sort by index
    perfromance = np.array(perfromance)  # (N,3)


    np.save(saving_path, {
        "latents": all_latent,
        "shapes": all_shapes,
        "performances": perfromance
    })
    print("creating the results finished !!!")
    print(rf"savinng in path {saving_path}")
