import os, math, uuid, sys, random
import numpy as np
import os, os.path
import re
def manual_save_func():
    Data_path = "./results/"
    num_files = len([name for name in os.listdir(Data_path) if os.path.isfile(os.path.join(Data_path, name))])
    cd = []
    cl = []

    files_list = [f for f in os.listdir(Data_path) if os.path.isfile(os.path.join(Data_path, f))]
    files_list.sort()
    os.chdir(Data_path)
    numbers_array = []
    
    for file in files_list:
        print(file)
        match = re.search(r'\d+', file)
        if match:
            number_part = match.group()
            numbers_array.append(int(number_part))
        with open(file) as f:
            cl = np.append(cl, float(next(f).split()[0]))
            cd = np.append(cd, float(next(f).split()[0]))


    cl = np.array(cl)
    cd = np.array(cd)

    final = np.append(np.expand_dims(cl, axis=1), np.expand_dims(cd, axis=1), axis=1)
    final = np.append(final, np.expand_dims(numbers_array, axis=1), axis=1)
        # Get a list of all files in the directory
    # files = os.listdir(Data_path)
    # Iterate through the list of files and remove each one
    # for file in files:
        # os.remove(os.path.join(Data_path, file))
    return final
    
all_cl_cd= manual_save_func()
print(all_cl_cd)


    
