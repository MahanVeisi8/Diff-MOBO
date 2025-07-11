import scipy.io as sio
import numpy as np
import subprocess
import os

def run_next_gen_dset(n_iter):
    #Edit the bash files to run with the correct iter number
    for i in range(3):
        sh_file_path = 'run_new_dset_%d.sh' %(i+1)
        new_line = "iter_n=%d" %(n_iter)
        with open(sh_file_path, "r") as sh_file:
            lines = sh_file.readlines()
        lines[2] = new_line + "\n"
        with open(sh_file_path, "w") as sh_file:
            sh_file.writelines(lines)
    
    # define three commands to run
    # Note that before this step you need to make sure that all three containers' id are known 
    # command1 = "ssh nansari@puck docker exec af0194da55a5 ./home/airfoil_UANA/run_new_dset_1.sh"
    # command2 = "ssh nansari@dione docker exec 5f9563809000 ./home/airfoil_UANA/run_new_dset_2.sh"
    # command3 = "ssh nansari@carpo docker exec 48734b2351e7 ./home/airfoil_UANA/run_new_dset_3.sh"
    command1 = "docker exec 78257c051b5b ./home/airfoil_UANA/run_rand_dset_1.sh"
    command2 = "docker exec 78257c051b5b ./home/airfoil_UANA/run_rand_dset_2.sh"
    command3 = "docker exec 78257c051b5b ./home/airfoil_UANA/run_rand_dset_3.sh"
    # run commands in parallel
    processes = [subprocess.Popen(cmd, shell=True) for cmd in [command1, command2, command3]]

    # wait for all processes to finish
    for process in processes:
        process.wait()

    print('new data set generation done')


    # Organizing the data
    # Initialize empty arrays to store the data
    latent_all = []
    performance_all = []

    # Loop through the three batches
    for n_batch in range(1, 4):
        # Read the design dataset
        dset_design_path = f'./Dataset/temp/dset_design_{n_batch}_{n_iter}.mat'
        dset_design =sio.loadmat(dset_design_path)
        latent = dset_design['latent']

        # # Read the performance dataset
        dset_perform_path = f"./Dataset/temp/dset_perform_{n_batch}_{n_iter}.mat"
        dset_perform = sio.loadmat(dset_perform_path)
        performance = dset_perform['performance']

        # Stack the arrays
        latent_all.append(latent)
        performance_all.append(performance)

    # Concatenate the stacked arrays along the first axis
    latent_all = np.concatenate(latent_all, axis=0)
    performance_all = np.concatenate(performance_all, axis=0)
    # Save the latent data in order to train the constraint detector 
    vaiolation_flag = (performance_all[:,0]!=-1000)
    data = {'latent_all':latent_all, 'vaiolation_flag':vaiolation_flag}
    sio.savemat('Dataset/latent_%d.mat' %(n_iter+1), {'dset': data})
    # clean the data of the failed cases
    latent_all = latent_all[performance_all[:,0]!=-1000,:]
    performance_all = performance_all[performance_all[:,0]!=-1000,:]
    data = {'X':latent_all, 'Y':performance_all}
    sio.savemat('Dataset/dset_%d.mat' %(n_iter+1), {'dset': data})


    #Remove the temp files
    folder_path = "./Dataset/temp"
    # Get a list of all files in the folder
    file_list = os.listdir(folder_path)
    # Loop through the list and delete each file
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        os.remove(file_path)