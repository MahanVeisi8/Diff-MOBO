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

# docker_mount_path = "/home/bardiya/projects/diffusion_air_manifolding/codes/creative-generativeai-diffusion/src/OpenFoam"
# docker_mount_path = "/home/bardiya/projects/diffusion_air_manifolding/codes/Airfoil_MPI_system"
docker_mount_path = "src/OpenFoam"
NUM_TO_GENERATE = 10000
BATCH_SIZE = 128
# NUM_TO_GENERATE = 2
# BATCH_SIZE = 2
docker_container_id = "f897792b6b56" 
saving_path = rf"src/optimization_loop/run_results.npy"
device = "cpu"
# device = "cuda"

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

    # Same architecture as in training
    model = Unet1D(
        dim=32,
        dim_mults=(2, 4, 8, 16),
        channels=2,  # X and Y
        dropout=0.1
    ).to(device)  # or .to(device)

    # Create the same diffusion wrapper
    diffusion = GaussianDiffusion1D(
        model,
        seq_length=192,      # must match your training setup
        objective='pred_noise',
        timesteps=1000,
        auto_normalize=False
    ).to(device)  # or .to(device)

    # Load checkpoint
    checkpoint_path = rf"src/diffusion_notebooks/DIffusion_model_weigths_and_datas/dpp_0.1_autonorm_true_125_from_base_ddpm/model_epoch_124.pt"
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

    np.save(os.path.join(docker_mount_path , "DB2.npy") , {
        "latents": np.vstack(all_latent),
        "shapes": np.vstack(all_shapes),
        "performances": None
    })
    print("================================")
    print(rf"starting openfoam in mount path {docker_mount_path} with docker container {docker_container_id}")
    command = (
        f'docker exec {docker_container_id} bash -c "'
        f'source /opt/openfoam5/etc/bashrc && '
        f'cd /home/airfoil_UANA && '
        f'python3 performance_finding.py"'
    )

    os.system(command)
    print("================================")

    performance_path =os.path.join(docker_mount_path , rf"performance.npy")
    perfromance = np.load(performance_path,allow_pickle=True)
    np.save(saving_path, {
        "latents": np.vstack(all_latent),
        "shapes": np.vstack(all_shapes),
        "performances": perfromance
    })
    print("creating the results finished !!!")
    print(rf"savinng in path {saving_path}")
