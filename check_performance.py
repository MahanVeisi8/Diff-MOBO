import numpy as np

performance_path = 'src/OpenFoam/performance.npy'
performance = np.load(performance_path,allow_pickle=True)
print(performance)
print(performance.shape)