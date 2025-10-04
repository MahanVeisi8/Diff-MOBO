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
import scipy.io as sio
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from utils import  BO_surrogate_uncertainty
import yaml


def inv_coords(xs_s):                   # xs_s shape (...,2,192) tensor
    xs_np = xs_s.permute(0,2,1).cpu().numpy()    # -> (B,192,2)
    # xs_np[...,0] = xs_np[...,0]*(x_max-x_min) + x_min
    # xs_np[...,1] = xs_np[...,1]*(y_max-y_min) + y_min
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

def GEN_UA(diffusion,device ,num_cores, number_iter = 0,number_generations=100 , population_size = 1000 , from_DB_innerloop = True):
    # n_iter = 2
    print('calculating surrogate pareto ...')
    if from_DB_innerloop:
        DB_innerloop = np.load(os.path.join("Database" , "DB_innerloop.npy"),allow_pickle=True).item()
        full_samples = DB_innerloop["latents"]  # (batch , 2,  192)
        
        print(full_samples.shape)
    problem_uncertainty = BO_surrogate_uncertainty(diffusion = diffusion,device=device,num_cores=num_cores,n_iter=number_iter)
    algorithm = NSGA2(pop_size=population_size)
    res = minimize(problem_uncertainty,
                algorithm,
                ('n_gen', number_generations),
                seed=1,
                verbose=False,
                X = full_samples.reshape(full_samples.shape[0], -1)  if from_DB_innerloop else None)
    Paretoset_uncertainty = res.X
    Out_surrogate_uncertainty = res.F
    # sio.savemat('surrogate_pareto/ParetoSet_test.mat' , {'ParetoSet': np.array(Paretoset_uncertainty)})
    # sio.savemat('surrogate_pareto/Out_surrogate_test.mat', {'Out_surrogate': np.array(Out_surrogate_uncertainty)})
    print(f"number of last generation sample is {len(Paretoset_uncertainty)}")
    np.save(os.path.join("Database" , "DB_NSGA.npy"),{
        'ParetoSet': np.array(Paretoset_uncertainty), # the latents
        'Out_surrogate': np.array(Out_surrogate_uncertainty)
    })

    return problem_uncertainty

def NSGA_latent_to_shape(model ,diffusion,num_cores, docker_mount_path, checkpoint_path, BATCH_SIZE=128):
    # Load NSGA latent vectors
    DB_NSGA = np.load(os.path.join("Database", "DB_NSGA.npy"), allow_pickle=True).item()
    NSGA_latent = DB_NSGA["ParetoSet"].reshape(DB_NSGA["ParetoSet"].shape[0], 2, -1)
    NSGA_latent = torch.from_numpy(NSGA_latent).float()

    # Load model weights
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.eval()
    model.to(device)
    print("Loaded model weights from:", checkpoint_path)

    # Create DataLoader for batching
    dataset = TensorDataset(NSGA_latent)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_cores-1)

    all_latent = []
    all_shapes = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            latent_batch = batch[0].to(device)  # shape: (B, 2, 192)
            samples = diffusion.latent_sample(latent_batch, is_ddim=True)
            generated_real = inv_coords(samples)  # assumes inv_coords is defined globally

            all_latent.append(latent_batch.cpu().numpy())
            all_shapes.append(generated_real)

            print(f"Processed {(i+1)*BATCH_SIZE}/{len(NSGA_latent)} latents")

    # Save results
    np.save(os.path.join(docker_mount_path, "DB_CFD.npy"), {
        "latents": np.vstack(all_latent),
        "shapes": np.vstack(all_shapes),
        "performance": []
    })
    print(f"Saved converted shapes to {os.path.join(docker_mount_path, 'DB_CFD.npy')}")

def CFD_simulation(docker_container_id):
    # command = fr"docker exec {docker_container_id} python3 /home/airfoil_UANA/performance_finding.py"
    command = (
        f'docker exec {docker_container_id} bash -c "'
        f'source /opt/openfoam5/etc/bashrc && '
        f'cd /home/airfoil_UANA && '
        f'python3 innerloop_performance_finding.py"'
    )
    os.system(command)

def Tagging_phase(docker_mount_path, iteration  = 0):
    """
        taggin the DB_CFD.npy and make it valid and invalid DB inn the Database directory
        retraining the UA_surrogates and others.
    """
    DB_CFD = np.load(os.path.join( docker_mount_path, "DB_CFD.npy"),allow_pickle=True).item()
    performance = DB_CFD["performance"]
    latents = DB_CFD["latents"]
    shapes = DB_CFD["shapes"]
    # print(f"{latents.shape=}")
    # print(f"{shapes.shape=}")
    # print(f"{performance.shape=}")
    valids = {
        "latents" : [],
        "shapes" : [],
        "performance" : []
    }
    invalids = {
        "latents" : [],
        "shapes" : [],
        "performance" : []
    }
    for i in range(len(latents)):
        if performance[i,0] == -1000:
            invalids["latents"].append(latents[i])
            invalids["shapes"].append(shapes[i])
            invalids["performance"].append(performance[i])
        else:
            valids["latents"].append(latents[i])
            valids["shapes"].append(shapes[i])
            valids["performance"].append(performance[i])

    # sttackig the np arrays
    if len(valids["latents"]) > 0:
        valids["latents"] = np.vstack(valids["latents"])
        valids["shapes"] = np.vstack(valids["shapes"])
        valids["performance"] = np.vstack(valids["performance"])
    
    if len(invalids["latents"]) > 0:
        invalids["latents"] = np.vstack(invalids["latents"])
        invalids["shapes"] = np.vstack(invalids["shapes"])
        invalids["performance"] = np.vstack(invalids["performance"])

    
    # print(f'{invalids["performance"].shape=}')
    # print(f'{valids["performance"].shape=}')
    
    np.save(os.path.join("Database" , f"DB_valids_iter_{iteration}.npy"),valids)
    np.save(os.path.join("Database" , f"DB_invalids_iter_{iteration}.npy"),invalids)

    # Appending valids to the DB_innerloop
    DB_innerloop = np.load(os.path.join("Database","DB_innerloop.npy"),allow_pickle=True).item()
    if len(valids["latents"]) > 0:
        if valids["latents"].dim() == 2:
            appending_latents = np.expand_dims(valids["latents"],axis = 0)
            appending_shapes = np.expand_dims(valids["shapes"],axis = 0)
        
        DB_innerloop["latents"] = np.concatenate([DB_innerloop["latents"] , appending_latents],axis = 0)
        DB_innerloop["shapes"] = np.concatenate([DB_innerloop["shapes"] , appending_shapes],axis = 0)
    
    
    print(f"saving the valids and invalids in Database")

def Retraining_UA_modules(model , checkpoint_path ,num_cores, batch_size = 128,epoches = 20,patience=5,lr = 1e-6 , iteration = 0):
    DB_valids = np.load(os.path.join("Database", f"DB_valids_iter_{iteration}.npy"),allow_pickle=True).item()
    if len(DB_valids["shapes"]) ==  0:
        print("No valid samples for this iteration passing the retraining ...")
        return 
    
    airfoils = torch.from_numpy(DB_valids["shapes"]).float()  # (batch, 192,2)
    airfoils = airfoils.reshape(airfoils.shape[0],-1) # (batch, 384)
    scores = torch.from_numpy(DB_valids["performance"]).float()[:,:2] #(batch,2)  -> (cl,cd)
    # Dataset
    dataset = TensorDataset(airfoils, scores)
    
    # Loss + Optimizer
    criterion = nn.L1Loss()   # L1 loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size,num_workers=num_cores-1, shuffle=True)

    losses = []
    best_loss = float("inf")
    patience_counter = 0
    for epoch in range(epoches):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epoches}", leave=False)
        total_loss = 0
        
        for x, y in loop:
            x, y = x.to(device), y.to(device)

            # forward
            preds = model(x)[:,]
            preds = torch.stack(preds,dim=0)
            preds = torch.mean(preds,dim=0) # (batch , 2) -> cl , cl/cd
            preds[:,1] = preds[:,0] / (preds[:,1] + 1e-10) # (batch , 2) -> cl , cd
            loss = criterion(preds, y)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            avg_loss = total_loss / len(dataloader)
            losses.append(avg_loss)
            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1}/{epoches} | Avg Loss: {total_loss/len(dataloader):.4f}")    
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0

            # Save best checkpoint
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(losses , os.path.join(checkpoint_path , f"losses_iter_{iteration}.pt"))
            torch.save(model.state_dict(), os.path.join(checkpoint_path , f"UA_weigths_iter_{iteration}.pt"))
            print(f"Saved new best model at epoch {epoch+1} with loss {best_loss:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered!")
            break

    # Retraining the Constraint Handler

    print(f"End retraining the modules")



if __name__== "__main__":

    # reading the yaml file
    with open("innerloop_2_config.yaml", "r") as file:
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
    device = config["model"]["device"]
    Unet_checkpoint_path = config["model"]["unet_checkpoint"]
    saving_path = rf"src/optimization_loop/Inner_loop/Database/DB_innerloop.npy"
    number_generations = config["genetic_algorithm"]["number_generations"]
    population_size = config["genetic_algorithm"]["population_size"]
    from_DB_innerloop = config["genetic_algorithm"]["from_DB_innerloop"]

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
    for iter in range(iterations):
        print(f"iteration: {iter}")
        # Stage 1:
        print("Stage 1")
        NSGA_BO_surrogate_modules = GEN_UA(diffusion=diffusion, 
                                    device=device,
                                    num_cores = num_cores, 
                                    number_generations=number_generations, 
                                    population_size=population_size,
                                    from_DB_innerloop=True)

        # Stage 2:
        print("Stage 2")
        NSGA_latent_to_shape(Unet_model , 
                    diffusion , 
                    num_cores,
                    BATCH_SIZE=BATCH_SIZE,
                    checkpoint_path=Unet_checkpoint_path,
                    docker_mount_path=docker_mount_path
                    )

        # Stage 3 (first  start it):
        print("Stage 3")
        CFD_simulation(docker_container_id=docker_container_id)

        # Stage 4:
        print("Stage 4")
        Tagging_phase(docker_mount_path , iteration=iter)

        # Stage 5: (retraining the UA_surrogate models [and possibly the constraint handler to])
        print("Stage 5")
        checkpoint_path = "Retraining_modules"
        Retraining_UA_modules(iteration=iter,
                            num_cores=num_cores,
                            model=NSGA_BO_surrogate_modules.UA_surrogate_model,
                            checkpoint_path=checkpoint_path)
    print("End  Inner loop procedure")
