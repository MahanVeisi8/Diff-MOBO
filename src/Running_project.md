# Overview and Structure Treefo the project
The main codes for running the projects are depicted in `optimization_loop` and train_diffusion
and `Inner_loop` directories.
``` 
.
├─ src/
│   ├─ diffusion_core/          # Diffusion model (datasets, model, diffusion, utils)
│   ├─ OpenFoam/                # CFD coupling and runner scripts (keep as-is)
│   ├─ optimization_loop/       # Outer loop orchestrator
│   │   ├─ Innerloop/
│   │   ├─ train_diffusion/
│   │   ├─ Outer_loop/
│   │   └─ README.md
│   ├─  surrogate_models/        # Surrogate models for fast evaluation
│   └─ Running_project.md                
│
├─ artifacts/                   # Run outputs, weights, results (Git LFS)
├─ assets/                      # Figures, logos, diagrams
├─ run_inner_loop.sh            #important for project running
├─ run_train_diffusion.sh       #important for project running
├─ .gitignore
└─ README.md (this file)
```

# Running Procedures:
The Link of the models and .pt and .npy file is in [Here](https://drive.google.com/drive/folders/1YMBxlG-IxpO_EQ71j8eP2Pcd2PpGexzR?dmr=1&ec=wgc-drive-globalnav-goto).
Also before running make sure that the OpenFOAM changes on the `ShapeToPerformance.py` file in the `Airfoil_simulation_1`
has not been set for the MPI clusters, please make sure that this file is the same as your edit.
## Stage1: Getting sudo access for running bash files

```sh
sudo chmod +x run_inner_loop.sh
sudo chmod +x run_train_diffusion.sh
```
## Stage2: Training Diffusion model on GPU:
The `run_train_diffusion.sh` is  responsible for running the `src/optimization_loop/train_diffusion/train_diffusion.py`.
first we have to update this sh file to run the codes in a GPU cluster.  
in the `src/optimization_loop/train_diffusion/`  directory there is a yaml file called `diffusion_config.yaml` which has the Hyperparameters and all configurations for the projects running.
the `src/optimization_loop/train_diffusion/weigths/` directory is for saving the diffusion models (and surrogate_model in case you want!) loggs and .pt files.
The yaml file has the these parameters:
We only need to set the inputs paths like:

-   `Unet_starting_path`: the path for the  diffusion model (for this only use the `model_epoch_499_RMSnorm_unscaled.pt`)
-   `BASE_DATA_DIR`: the path of the data director in the base root of the project
-   `xs_train_path`: the path of xs_train.npy  file in the BASE_DATA_DIR
-   `ys_train_path`:  the path of ys_train.npy  file in the BASE_DATA_DIR
-   `coord_min_max_path`: the path of coord_min_max.npy  file in the BASE_DATA_DIR
-   `label_min_max_path`: the path of label_min_max.npy  file in the BASE_DATA_DIR
-   The `init_cl_path`: for setting up each pipeline weigths for the surrogate_model for cl_weigthts.
-   The `init_cd_path`: for setting up each pipeline weigths for the surrogate_model for cd_weigthts.
-   `all_weigths_path`: the path for the weigths of Ensemble if all of them is ina .pt file.
-   `number_of_cores`: number of cores for running the codes.

## Stage3. Training The Innerloop  procedure.
The `run_inner_loop.sh` is responsible for running the codes of `src/optimization_loop/Inner_loop/innerloop_creation_i.py`. 
First we have to change this code to only run the `python3 innerloop_creation_3.py --iteration "$i"` on Docker and CPU(no need for GPU!) and rest on GPU cluster in the bash format (i dont know but ssh is good for these things if we have  access to clusters).
After running the codes we have the `src/optimization_loop/Inner_loop/UA_surrogate_weights/updated` path for the UA_srurrogate_model weights. Also, `src/optimization_loop/Inner_loop/Database` is responsible for gathering the DB's and the results file which are `DB_valids_iter_i.npy` and `DB_invalids_iter_i.npy`.
Also in the `src/OpenFoam/` we added one file, `src/OpenFoam/innerloop_performance_finding.py` and `src/OpenFoam/innerloop_init_performance_finding.py` which is responsible for running the airfoil shapes specificly for the innerloop codes and is super   important to set the `NUM_CORE` in there.
The yaml file `src/optimization_loop/Inner_loop/innerloop_config.yaml` is responsible for Hyperparams and configurations path.
the things we have to set here is:
But be carefull:
**The path's here should start with `../../..`.**
-   `docker_container_id`: the id of the docker container  of the openfoam ( make sure it has been started before running!)
-   `unet_checkpoint`: the path of the diffusion model's weigth that has been trained in the first stage 
-   `[sampling][number_to_generate]`: how many number do we want to generate in the prestaging phases (typiically 1000).
-   `[sampling][batch_size]`: for the prestaging phase, better be 128 or 256.
-   `[process][num_cores]`: set the number of cores.
-   `[genetic_algorithm][number_generations]`: number of generations (typically 100).
-   `[genetic_algorithm][population_size]`: number of population_size for each  generation (typically 1000).
-   `[genetic_algorithm][from_DB_innerloop]`: wheter to start exploring near the DB_innerloop samples first(typically true).
-   `[UA_surrogate_model][saved_update_path]`: the file path that saves the final weigth of the surrogate model. 


after running this we should have `src/optimization_loop/Inner_loop/UA_surrogate_weights/updated/UA_surrogate_model_8_channel.pt` for UA_weigths and the `DB_valids` and `DB_invvalids`  for each iterations.

At the End, we need to save the `DB_innerloops.npy` files for the correct samples. 
