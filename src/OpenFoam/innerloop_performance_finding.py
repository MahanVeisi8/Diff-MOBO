import scipy.io as sio
import numpy as np
# from tf_analytic_airfoil import airfoil_fun
from Airfoil_simulation_1.ShapeToPerformance import shape_to_performance as STP1
from Airfoil_simulation_2.ShapeToPerformance import shape_to_performance as STP2
from Airfoil_simulation_3.ShapeToPerformance import shape_to_performance as STP3
import argparse
import os,sys

NUM_CORES = 2

DB = np.load(rf"DB_CFD.npy",allow_pickle=True).item()
airfoil_shape = DB["shapes"]
DB_p = DB["performance"]

total_shapes = len(airfoil_shape)
done = 0
all_performances = []
while done < total_shapes:
    cur = min(NUM_CORES, total_shapes - done)
    batch = airfoil_shape[done:done+cur , :,:]
    performance = STP1(batch)
    print(performance)
    performance[:2] = performance[:2] + done
    all_performances.append(performance)
    done += cur


np.save("DB_CFD.npy" , {
    "latents" : DB["latents"],
    "shapes" : DB["shapes"],
    "performance" : np.vstack(all_performances),
})

print("CFD simulation done!!!")