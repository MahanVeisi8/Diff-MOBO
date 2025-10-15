# import scipy.io as sio
# import numpy as np
# # from tf_analytic_airfoil import airfoil_fun
# from Airfoil_simulation_1.ShapeToPerformance import shape_to_performance as STP1
# from Airfoil_simulation_2.ShapeToPerformance import shape_to_performance as STP2
# from Airfoil_simulation_3.ShapeToPerformance import shape_to_performance as STP3
# import argparse
# import os,sys
# import pickle

# """
#     please make sure to copy and paste  openFOAM_i,
#     200 times and rename them (like in the preiouse stages)
# """

# NUM_CORES = 200

# # DB = np.load(rf"DB_CFD.npy",allow_pickle=True).item()
# with open("DB_init.npy", "rb") as f:
#     DB = pickle.load(f)

# airfoil_latent = DB["latents"]
# airfoil_shape = DB["shapes"]
# print(airfoil_shape.shape)
# total_shapes = len(airfoil_shape)
# done = 0
# all_performances = []
# all_latents = []
# all_shapes = []

# while done < total_shapes:
#     cur = min(NUM_CORES, total_shapes - done)
#     batch = airfoil_shape[done:done+cur , :,:]
#     batch_latent = airfoil_latent[done:done+cur , :,:]
#     performance, latents, shapes = STP1(batch, batch_latent)
#     print(performance)
#     performance[:, 2] = performance[:, 2] + done
#     all_performances.append(performance)
#     all_latents.append(latents)
#     all_shapes.append(shapes)
#     done += cur

# p = np.vstack(all_performances)
# l = np.vstack(all_latents)
# s = np.vstack(all_shapes)
# print(p.shape, l.shape, s.shape)

# # Get indices that would sort the 3rd column (index 2)
# indices = np.argsort(p[:, 2], axis=0)
# print(indices)

# # Use those indices to reorder p
# p = p[indices]
# l = l[indices]
# s = s[indices]
# print(p)

# np.save("DB_init.npy" , {
#     "latents" : l,
#     "shapes" : s,
#     "performance" : p,
# })
# print("CFD simulation done!!!")



import scipy.io as sio
import numpy as np
# from tf_analytic_airfoil import airfoil_fun
from Airfoil_simulation_1.ShapeToPerformance2 import shape_to_performance as STP1
from Airfoil_simulation_2.ShapeToPerformance import shape_to_performance as STP2
from Airfoil_simulation_3.ShapeToPerformance import shape_to_performance as STP3
import argparse
import os,sys
import pickle

"""
    please make sure to copy and paste  openFOAM_i,
    200 times and rename them (like in the preiouse stages)
"""

NUM_CORES = 200

# DB = np.load(rf"DB_init.npy",allow_pickle=True).item()
# with open("run_results_inSTP.npy", "rb") as f:
#     DB2 = pickle.load(f)[:5]
with open("DB_init.npy", "rb") as f:
    DB = pickle.load(f)

airfoil_latent = DB["latents"]
airfoil_shape = DB["shapes"]
total_shapes = len(airfoil_shape)

done = 0
all_performances = []
all_latents = []
all_shapes = []

batch_nums = (np.arange(np.ceil(total_shapes/NUM_CORES)) * NUM_CORES).tolist()

batch_nums.append(total_shapes)
# print("hiii")
# print(batch_nums)
# # print(total_shapes)
# sys.exit()
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

# np.save("performance.npy" , np.vstack(all_performances))
np.save("DB_init.npy", {
        "latents": l,
        "shapes": s,
        "performance": p
    })
print("CFD simulation done!!!")