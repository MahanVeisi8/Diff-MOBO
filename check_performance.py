import numpy as np

performance_path = 'src/OpenFoam/performance.npy'
result_path = 'src/optimization_loop/run_results.npy'
DB2_path = 'src/OpenFoam/DB2.npy'
result_new = 'src/OpenFoam/run_results_inSTP.npy'
performance = np.load(performance_path,allow_pickle=True)
results = np.load(result_path,allow_pickle=True).item()
results_new = np.load(result_new,allow_pickle=True).item()
print(results['performances'][:, 2])
print(results_new['performances'][:, 2])
print(results_new['performances'].shape)
print(results_new['latents'].shape, results_new['shapes'].shape)
# Performance is CL, CD, Index
# Results is CL, CD, Index