import numpy as np
import pickle

DB_path = 'src/OpenFoam/xs_train.npy'
pseudo_db_path = 'src/OpenFoam/DB2_xstrain.npy'
result_path_stp = 'src/OpenFoam/results/run_results_inSTP_xstrain.npy'
result_path_stp = 'src/optimization_loop/run_results_xstrain.npy'

# db = pickle.load(open(DB_path, "rb"))
results = np.load(result_path_stp, allow_pickle=True).item()
shapes_in_results = results['shapes']
shapes_in_results = np.array(shapes_in_results).reshape(-1, 192, 2)
shapes_in_pseudo_db = np.load(pseudo_db_path, allow_pickle=True)['shapes']
shapes_in_db = np.load(DB_path, allow_pickle=True)

# compare if pseudo_db shapes match db shapes
if np.array_equal(shapes_in_db[:len(shapes_in_pseudo_db)], shapes_in_pseudo_db):
    print("The shapes in pseudo_db match the shapes in db.")
else:
    print("The shapes in pseudo_db do not match the shapes in db.")

if np.array_equal(shapes_in_db[:len(shapes_in_results)], shapes_in_results):
    print("The shapes in results match the shapes in db.")
else:
    print("The shapes in results do not match the shapes in db.")

# print the performances for the first 5 shapes in shapes_in_db
print(len(shapes_in_results))
for i in range(0, len(shapes_in_db), 1000):
    shape = shapes_in_db[i]
    for j, s in enumerate(shapes_in_results):
        if np.array_equal(shape, s):
            print(f"Shape {i} found in results with performance: {results['performances'][j]}")
            break

print(results['performances'][:5])  # should be 0,1,2,3,...
