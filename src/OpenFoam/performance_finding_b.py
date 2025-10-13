# # # import scipy.io as sio
# # # import numpy as np
# # # # from tf_analytic_airfoil import airfoil_fun
# # # from Airfoil_simulation_1.ShapeToPerformance import shape_to_performance as STP1
# # # from Airfoil_simulation_2.ShapeToPerformance import shape_to_performance as STP2
# # # from Airfoil_simulation_3.ShapeToPerformance import shape_to_performance as STP3
# # # import argparse
# # # import os,sys
# # # import pickle 
# # # import traceback

# # # # change the num cores after copy ans pasting its  really important
# # # NUM_CORES = 200

# # # # DB2 = np.load(rf"src/OpenFoam/DB2.npy",allow_pickle=True).item()
# # # with open("DB2.npy", "rb") as f:
# # #     DB2 = pickle.load(f)
# # # airfoil_shape = DB2["shapes"]
# # # print(airfoil_shape.shape)
# # # total_shapes = len(airfoil_shape)
# # # done = 0
# # # all_performances = []
# # # while done < total_shapes:
# # #     cur = min(NUM_CORES, total_shapes - done)
# # #     batch = airfoil_shape[done:done+cur , :,:]
# # #     try:
# # #         performance = STP1(batch)
# # #     except Exception as e:
# # #         print("Exception in worker process:")
# # #         traceback.print_exc()
# # #         raise
# # #     # # performance = STP1(batch)
# # #     print(performance)
# # #     all_performances.append(performance)
# # #     done += cur
# # # np.save("performance.npy" , np.vstack(all_performances))
# # # print("done!!!")

import scipy.io as sio
import numpy as np
# from tf_analytic_airfoil import airfoil_fun
from Airfoil_simulation_1.ShapeToPerformance import shape_to_performance as STP1
from Airfoil_simulation_2.ShapeToPerformance import shape_to_performance as STP2
from Airfoil_simulation_3.ShapeToPerformance import shape_to_performance as STP3
import argparse
import os,sys
import pickle


NUM_CORES = 200

with open("DB2_xstest.npy", "rb") as f:
    DB2 = pickle.load(f)
airfoil_latent = DB2["latents"]
airfoil_shape = DB2["shapes"]
print(airfoil_shape.shape)
total_shapes = len(airfoil_shape)
done = 0
all_performances = []
all_latents = []
all_shapes = []
while done < total_shapes:
    cur = min(NUM_CORES, total_shapes - done)
    batch = airfoil_shape[done:done+cur , :,:]
    batch_latent = airfoil_latent[done:done+cur , :,:]
    performance, latents, shapes = STP1(batch, batch_latent)
    print(performance)
    performance[:, 2] = performance[:, 2] + done
    all_performances.append(performance)
    all_latents.append(latents)
    all_shapes.append(shapes)
    done += cur
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

np.save("performance.npy" , np.vstack(all_performances))
np.save("run_results_inSTP_xtest.npy", {
        "latents": l,
        "shapes": s,
        "performances": p
    })
print("done!!!")