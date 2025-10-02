import scipy.io as sio
import numpy as np
# from tf_analytic_airfoil import airfoil_fun
from Airfoil_simulation_1.ShapeToPerformance import shape_to_performance as STP1
from Airfoil_simulation_2.ShapeToPerformance import shape_to_performance as STP2
from Airfoil_simulation_3.ShapeToPerformance import shape_to_performance as STP3
import argparse
import os,sys
import pickle 
import traceback

# change the num cores after copy ans pasting its  really important
NUM_CORES = 200

# DB2 = np.load(rf"src/OpenFoam/DB2.npy",allow_pickle=True).item()
with open("DB2.npy", "rb") as f:
    DB2 = pickle.load(f)
airfoil_shape = DB2["shapes"]
print(airfoil_shape.shape)
total_shapes = len(airfoil_shape)
done = 0
all_performances = []
while done < total_shapes:
    cur = min(NUM_CORES, total_shapes - done)
    batch = airfoil_shape[done:done+cur , :,:]
    try:
        performance = STP1(batch)
    except Exception as e:
        print("Exception in worker process:")
        traceback.print_exc()
        raise
    # # performance = STP1(batch)
    print(performance)
    all_performances.append(performance)
    done += cur
np.save("performance.npy" , np.vstack(all_performances))
print("done!!!")
