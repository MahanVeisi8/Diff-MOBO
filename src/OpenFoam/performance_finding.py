import scipy.io as sio
import numpy as np
# from tf_analytic_airfoil import airfoil_fun
from Airfoil_simulation_1.ShapeToPerformance import shape_to_performance as STP1
from Airfoil_simulation_2.ShapeToPerformance import shape_to_performance as STP2
from Airfoil_simulation_3.ShapeToPerformance import shape_to_performance as STP3
import argparse
import os,sys


NUM_CORES = 2

DB2 = np.load(rf"DB2.npy",allow_pickle=True).item()
airfoil_shape = DB2["shapes"]
print(airfoil_shape.shape)
total_shapes = len(airfoil_shape)
done = 0
all_performances = []
l = 0
while done < total_shapes:
    if l> 5:
        break
    cur = min(NUM_CORES, total_shapes - done)
    batch = airfoil_shape[done:done+cur , :,:]
    performance = STP1(batch)
    print(performance)
    performance[:2] = performance[:2] + done
    all_performances.append(performance)
    done += cur
    l += 1
p = np.vstack(all_performances)
print(p.shape)

# Get indices that would sort the 3rd column (index 2)
indices = np.argsort(p[:, 2], axis=0)
print(indices)

# Use those indices to reorder p
p = p[indices]
print(p)

np.save("performance.npy" , np.vstack(all_performances))
print("done!!!")
