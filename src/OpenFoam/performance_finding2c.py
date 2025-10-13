# import scipy.io as sio
# import numpy as np
# # from tf_analytic_airfoil import airfoil_fun
# from Airfoil_simulation_1.ShapeToPerformance import shape_to_performance as STP1
# from Airfoil_simulation_2.ShapeToPerformance import shape_to_performance as STP2
# from Airfoil_simulation_3.ShapeToPerformance import shape_to_performance as STP3
# import argparse
# import os,sys
# import argparse


# NUM_CORES = 200

# parser = argparse.ArgumentParser()
# parser.add_argument("--num_batch", type=int, default=0,help="Path to the input file")        # positional
# parser.add_argument("--offset",default=200, type=int)  # optional

# args = parser.parse_args()
# num_batch = args.num_batch
# offset = args.offset


# DB2 = np.load(rf"DB2.npy",allow_pickle=True).item()
# index_offset = 200 * num_batch
# airfoil_shape = DB2["shapes"][index_offset:index_offset+offset,:,:]

# performance = STP1(airfoil_shape)
# performance[:,2] = performance[:,2] + 200 * num_batch


# # Get indices that would sort the 3rd column (index 2)
# indices = np.argsort(performance[:, 2], axis=0)
# print(indices)

# # Use those indices to reorder p
# p = performance[indices]
# print(p)

# np.save(f"performance_batch_{num_batch}.npy" , p)
# print("done!!!")



import scipy.io as sio
import numpy as np
# from tf_analytic_airfoil import airfoil_fun
from Airfoil_simulation_1.ShapeToPerformance2 import shape_to_performance as STP1
from Airfoil_simulation_2.ShapeToPerformance import shape_to_performance as STP2
from Airfoil_simulation_3.ShapeToPerformance import shape_to_performance as STP3
import argparse
import os,sys
import pickle


NUM_CORES = 200

# DB2 = np.load(rf"run_results_inSTP.npy",allow_pickle=True).item()
with open("DB2_xstrain.npy", "rb") as f:
    DB2 = pickle.load(f)

airfoil_latent = DB2["latents"]
airfoil_shape = DB2["shapes"]
total_shapes = len(airfoil_shape)
done = 0
all_performances = []
all_latents = []
all_shapes = []

batch_nums = (np.arange(np.ceil(total_shapes/NUM_CORES)) * NUM_CORES).tolist()
if total_shapes % NUM_CORES != 0:
    batch_nums.append(total_shapes)
# print("hiii")
# print(batch_nums)
indices = zip(batch_nums[:-1], batch_nums[1:])
# for a in indices:
#     print(a)
# sys.exit()
for index,(first_index , last_index) in enumerate(indices):
    first_index =int(first_index)
    last_index =int(last_index)
    print(f"fi:{first_index} , li:{last_index}")
    batch = airfoil_shape[first_index:last_index , :,:]
    batch_latent = airfoil_latent[first_index:last_index , :,:]
    performance, latents, shapes = STP1(batch, batch_latent,index)
    print(performance)
    performance[:, 2] = performance[:, 2] + first_index
    all_performances.append(performance)
    all_latents.append(latents)
    all_shapes.append(shapes)
print(all_performances)

p = np.vstack(all_performances)
l = np.vstack(all_latents)
s = np.vstack(all_shapes)
print(p.shape, l.shape, s.shape)

# Get indices that would sort the 3rd column (index 2)
indices = np.argsort(p[:, 2], axis=0)
print(indices)

# Use those indices to reorder p
p = p[indices]
l = l[indices]
s = s[indices]
print(p)

np.save("performance_C.npy" , np.vstack(all_performances))
np.save("run_results_inSTP_xtrain.npy", {
        "latents": l,
        "shapes": s,
        "performances": p
    })
print("done!!!")