import torch
import scipy.io as sio
import torch.nn as nn
import numpy as np
from pymoo.core.problem import Problem
from models import UA_surrogate_model, MultiLayerPerceptron_forward , MultiLayerPerceptron_forward_classifier
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset , random_split
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os, sys
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from tqdm import tqdm
import pickle
from scipy.stats import norm


DATA_DIR = Path(rf"../../../data")
coord_mm = np.load(DATA_DIR/"coord_min_max.npy")  # [[x_min,y_min],[x_max,y_max]]
x_min,y_min = coord_mm[0]; x_max,y_max = coord_mm[1]

class BO_surrogate_uncertainty(Problem):
    def __init__(self,diffusion,num_cores=2, device = "cuda" ,n_iter=0 ):
        super().__init__(n_var=384, n_obj=4, xl=0, xu=1)
        
        # setting up the diffusion models for getting designs out of latents
        self.diffusion = diffusion
        self.device = device
        self.num_cores= num_cores
        # Load the forward model    
        input_size = 384
        hidden_size_mu = [150, 200, 200 , 150]
        self.num_classes = 2
        
        # Load the models
        # self.mu_models = []
        # for n_net in range(1, 11):
        #     # Load mu models
        #     mu_model = MultiLayerPerceptron_forward(input_size, hidden_size_mu, self.num_classes, n_net)
        #     model_name = 'Models/iter_%d/mu_net_%d.ckpt' % (n_iter, n_net)
        #     mu_model.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))
        #     mu_model.eval()
        #     self.mu_models.append(mu_model)

        self.UA_surrogate_model = UA_surrogate_model().to(self.device)
        self.UA_surrogate_model.eval()
        # # if we need outo classifier
        # self.classifier_model = MultiLayerPerceptron_forward_classifier(input_size, hidden_size_mu, self.num_classes)
        # classifier_model_model_name = 'UA_surrogate_weights/Constraint_handlers/constraint_handler_%d.ckpt' % (n_iter)
        # a =  torch.load(classifier_model_model_name, map_location=torch.device('cpu'))
        # self.classifier_model.load_state_dict(torch.load(classifier_model_model_name, map_location=torch.device('cpu')))
        # self.classifier_model.eval()

    def _evaluate(self, latents, out, *args, **kwargs):
        net_n = len(self.UA_surrogate_model.cl_forward_mlps)
        # getting airfoil designs out of the latents
        latents = np.clip(latents, 1e-3, 1-1e-3)  # avoid infs
        latents = norm.ppf(latents)  # now ~ N(0,1)
        latents = torch.from_numpy(latents).float() # this latent has  been generated form the algorithm (batch , 384)
        # Assuming latents in [0,1]

        designs = self._latents_to_shapes(latents.reshape(latents.shape[0] , 2,  -1))

        # getting the UA infromations from the  airfoil designs 
        batchsize = designs.shape[0]
        reproduced_Performance_ensemble = torch.empty(net_n, batchsize, self.num_classes)
        reproduced_Performance_mu = torch.empty(batchsize, self.num_classes)
        uncertainty_epistemic = torch.empty((batchsize, self.num_classes))

        out_list = self.UA_surrogate_model(designs.reshape(batchsize, -1).to(self.device))
        reproduced_Performance_ensemble = torch.stack(out_list,dim=0).to("cpu")

        # calculating the mean  and  epistemic variance
        reproduced_Performance_mu = (1 / 10) * torch.sum(reproduced_Performance_ensemble, 0)
        uncertainty_epistemic = (1 / 10) * torch.sum(
            reproduced_Performance_ensemble ** 2 - reproduced_Performance_mu.repeat(net_n, 1, 1) ** 2,
            0)
        
        out["F"] = torch.cat((-reproduced_Performance_mu[:, :2].detach(), -uncertainty_epistemic[:, :2]), 1)
        out["F"] = out["F"].detach().numpy()

    def _latents_to_shapes(self, latents , BATCH_SIZE = 128):
        """
        Converts latents (batch, 2, 192) to generated shapes (batch, 192, 2)
        using self.diffusion.latent_sample and inv_coords.
        Uses DataLoader for CPU batching and optional parallelism.
        """
        if isinstance(latents, np.ndarray):
            latents = torch.from_numpy(latents).float().to(self.device)
        device = self.device
        num_to_generate = len(latents)
        batch_size      = BATCH_SIZE

        dataset = TensorDataset(latents)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=self.num_cores-1)

        all_shapes = []
        for i, batch in enumerate(loader):
            latent_batch = batch[0].to(self.device)  # shape: (B, 2, 192)
            print(f"{latent_batch.shape=}")
            with torch.no_grad():
                samples = self.diffusion.latent_sample(latent_batch, is_ddim=True)
                generated_real = self.inv_coords(samples)
                all_shapes.append(generated_real)
            print(f"Processed { (i+1)*batch_size } / { len(latents) } latents")

        # Stack all outputs
        return torch.from_numpy(np.vstack(all_shapes)).to("cpu")

    def inv_coords(self, xs_s):                     # xs_s shape (...,2,192) tensor
        xs_np = xs_s.permute(0,2,1).cpu().numpy()   # -> (B,192,2)
        xs_np[...,0] = xs_np[...,0]*(x_max-x_min) + x_min
        xs_np[...,1] = xs_np[...,1]*(y_max-y_min) + y_min
        return xs_np                                # (B,192,2) numpy
    
def inv_coords(xs_s):                   # xs_s shape (...,2,192) tensor
    xs_np = xs_s.permute(0,2,1).cpu().numpy()    # -> (B,192,2)
    # xs_np[...,0] = xs_np[...,0]*(x_max-x_min) + x_min
    # xs_np[...,1] = xs_np[...,1]*(y_max-y_min) + y_min
    return xs_np                                # (B,192,2) numpy

def init_generate_samples_latents(model ,diffusion, checkpoint_path,docker_mount_path, NUM_TO_GENERATE , BATCH_SIZE, device):
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

    data = {
        "latents": np.vstack(all_latent),
        "shapes": np.vstack(all_shapes),
        "performance": None
    }

    with open(os.path.join(docker_mount_path , "DB_init.npy"), "wb") as f:
        pickle.dump(data, f, protocol=4) # Compatible with python 3.6.9 in the docker

def init_CFD_simulation(docker_container_id):
    # command = fr"docker exec {docker_container_id} python3 /home/airfoil_UANA/performance_finding.py"
    command = (
        f'docker exec {docker_container_id} bash -c "'
        f'source /opt/openfoam5/etc/bashrc && '
        f'cd /home/airfoil_UANA && '
        f'python3 innerloop_init_performance_finding.py"'
    )
    os.system(command)

def init_tagging(docker_mount_path, iteration  = 0):
    """
        tagging the DB_init.npy and make it valid and invalid DB inn the Database directory
        retraining the UA_surrogates and others.
    """
    # with open(os.path.join( docker_mount_path, "DB_init.npy"), "rb") as f:
    #     DB_CFD = pickle.load(f)
    DB_CFD = np.load(os.path.join( docker_mount_path, "DB_init.npy"),allow_pickle=True).item()
    print(DB_CFD.keys())
    performance = DB_CFD["performance"]
    latents = DB_CFD["latents"]
    shapes = DB_CFD["shapes"]
    print(f"{latents.shape=}")
    print(f"{shapes.shape=}")
    print(f"{performance.shape=}")
    print(performance)
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
        valids["latents"] = np.stack(valids["latents"])
        valids["shapes"] = np.stack(valids["shapes"])
        valids["performance"] = np.stack(valids["performance"])
    
    if len(invalids["latents"]) > 0:
        invalids["latents"] = np.stack(invalids["latents"])
        invalids["shapes"] = np.stack(invalids["shapes"])
        invalids["performance"] = np.stack(invalids["performance"])

    # print(f"{valids['latents'].shape=}")
    # print(f"{valids['shapes'].shape=}")
    # print(f"{valids['performance'].shape=}")
    with open(os.path.join("Database" , "DB_innerloop.npy"), "wb") as f:
        pickle.dump(valids, f, protocol=4) # Compatible with python 3.6.9 in the docker

def GEN_UA(config,problem_uncertainty:BO_surrogate_uncertainty,diffusion,device ,num_cores, number_iter = 0,number_generations=100 , population_size = 1000 , from_DB_innerloop = True):
    # n_iter = 2
    print('calculating surrogate pareto ...')
    if from_DB_innerloop:
        # DB_innerloop = np.load(os.path.join("Database" , "DB_innerloop.npy"),allow_pickle=True).item()
        with open(os.path.join("Database" , "DB_innerloop.npy"), "rb") as f:
            DB_innerloop = pickle.load(f)
        full_samples = DB_innerloop["latents"]  # (batch , 2,  192)
        
        print(full_samples.shape)

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
    
    data = {
        'ParetoSet': np.array(Paretoset_uncertainty), # the latents
        'Out_surrogate': np.array(Out_surrogate_uncertainty)
    }
    with open(os.path.join("Database" , "DB_NSGA.npy"), "wb") as f:
        pickle.dump(data, f, protocol=4) # Compatible with python 3.6.9 in the docker   

    return problem_uncertainty

def NSGA_latent_to_shape(model ,diffusion,num_cores, docker_mount_path, checkpoint_path,device,BATCH_SIZE=128):
    # Load NSGA latent vectors
    # DB_NSGA = np.load(os.path.join("Database", "DB_NSGA.npy"), allow_pickle=True).item()
    with open(os.path.join("Database", "DB_NSGA.npy"), "rb") as f:
            DB_NSGA = pickle.load(f)
        
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


    data = {
        "latents": np.vstack(all_latent),
        "shapes": np.vstack(all_shapes),
        "performance": None
    }
    # Save results
    with open(os.path.join(docker_mount_path , "DB_CFD.npy"), "wb") as f:
        pickle.dump(data, f, protocol=4) # Compatible with python 3.6.9 in the docker

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
    # with open(os.path.join( docker_mount_path, "DB_CFD.npy"), "rb") as f:
    #         DB_CFD = pickle.load(f)

    performance = DB_CFD["performance"]
    latents = DB_CFD["latents"]
    shapes = DB_CFD["shapes"]
    print(f"{latents.shape=}")
    print(f"{shapes.shape=}")
    print(f"{performance.shape=}")
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
        valids["latents"] = np.stack(valids["latents"])
        valids["shapes"] = np.stack(valids["shapes"])
        valids["performance"] = np.stack(valids["performance"])
    
    if len(invalids["latents"]) > 0:
        invalids["latents"] = np.stack(invalids["latents"])
        invalids["shapes"] = np.stack(invalids["shapes"])
        invalids["performance"] = np.stack(invalids["performance"])

    
    # print(f'{invalids["performance"].shape=}')
    # print(f'{valids["performance"].shape=}')
    
    with open(os.path.join("Database" , f"DB_valids_iter_{iteration}.npy"), "wb") as f:
        pickle.dump(valids, f, protocol=4) # Compatible with python 3.6.9 in the docker
    with open(os.path.join("Database" , f"DB_invalids_iter_{iteration}.npy"), "wb") as f:
        pickle.dump(invalids, f, protocol=4) # Compatible with python 3.6.9 in the docker

    # Appending valids to the DB_innerloop
    # DB_innerloop = np.load(os.path.join("Database","DB_innerloop.npy"),allow_pickle=True).item()
    with open(os.path.join("Database","DB_innerloop.npy"), "rb") as f:
        DB_innerloop = pickle.load(f)
    print(f"{DB_innerloop['latents'].shape=}")
    print(f"{DB_innerloop['shapes'].shape=}")
    print(f"{DB_innerloop['performance'].shape=}")
    if len(valids["latents"]) > 0:
        if valids["latents"].dim() == 2:
            appending_latents = np.expand_dims(valids["latents"],axis = 0)
            appending_shapes = np.expand_dims(valids["shapes"],axis = 0)
        
        DB_innerloop["latents"] = np.concatenate([DB_innerloop["latents"] , appending_latents],axis = 0)
        DB_innerloop["shapes"] = np.concatenate([DB_innerloop["shapes"] , appending_shapes],axis = 0)
        DB_innerloop["performance"] = np.concatenate([DB_innerloop["performance"] , appending_shapes],axis = 0)

    with open(os.path.join("Database" , f"DB_innerloop.npy"), "wb") as f:
        pickle.dump(DB_innerloop, f, protocol=4) # Compatible with python 3.6.9 in the docker
    print(f"saving the valids and invalids in Database")

def Retraining_UA_modules(model, 
                          checkpoint_path,
                          num_cores, 
                          device,
                          batch_size=128,
                          epoches=200,
                          patience=50,
                          lr=1e-6, 
                          iteration=0,
                          LAMBDA_CL=1,
                          LAMBDA_CD=100):

    # Load valid samples
    with open(os.path.join("Database", f"DB_innerloop.npy"), "rb") as f:
        DB_valids = pickle.load(f)

    if len(DB_valids["shapes"]) == 0:
        print("No valid samples for this iteration passing the retraining ...")
        return 

    airfoils = torch.from_numpy(DB_valids["shapes"]).float()  # (batch, 192, 2)
    airfoils = airfoils.reshape(airfoils.shape[0], -1)        # (batch, 384)
    scores = torch.from_numpy(DB_valids["performance"]).float()[:, :2]  # (batch, 2) -> (cl, cd)
    print(f"{DB_valids['performance'].shape=}")
    print(f"{DB_valids['shapes'].shape=}")
    print(f"{DB_valids['latents'].shape=}")

    # Create dataset and split into train/validation sets
    dataset = TensorDataset(airfoils, scores)
    val_ratio = 0.2
    val_size = max(1,int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_cores-1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_cores-1, shuffle=False)
    print(f"{len(train_loader)=}")
    print(f"{len(val_loader)}")

    # Loss + Optimizer
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Tracking
    losses = []
    best_val_loss = float("inf")
    patience_counter = 0

    # Training loop
    for epoch in range(epoches):
        model.train()
        total_train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epoches}", leave=False)

        for x, y in loop:
            x, y = x.to(device), y.to(device)

            preds = model.get_cl_cd(x)
            preds[:, 0] *= 1000
            preds[:, 1] *= 1000
            y[:, 0] *= 1000
            y[:, 1] *= 1000

            loss_cl = criterion(preds[:, 0], y[:, 0])
            loss_cd = criterion(preds[:, 1], y[:, 1])
            loss = LAMBDA_CL * loss_cl + LAMBDA_CD * loss_cd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation phase
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)

                preds_val = model.get_cl_cd(x_val)
                
                val_loss_cl = criterion(preds_val[:, 0], y_val[:, 0])
                val_loss_cd = criterion(preds_val[:, 1], y_val[:, 1])
                val_loss = LAMBDA_CL * val_loss_cl + LAMBDA_CD * val_loss_cd

                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epoches} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # --- Early stopping based on validation loss ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            # Keep the saving method identical
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(losses, os.path.join(checkpoint_path, f"losses_iter_{iteration}.pt"))
            torch.save(model.state_dict(), os.path.join(checkpoint_path, f"UA_weigths_iter_{iteration}.pt"))
            torch.save(model.state_dict(), os.path.join(checkpoint_path, f"UA_surrogate_model_8_channel.pt"))
            print(f"Saved new best model at epoch {epoch+1} with val loss {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epoch(s).")

        if patience_counter >= patience:
            print("Early stopping triggered based on validation loss.")
            break

    print("End retraining the modules")
