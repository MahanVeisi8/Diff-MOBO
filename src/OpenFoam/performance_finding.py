import scipy.io as sio
import numpy as np
# from tf_analytic_airfoil import airfoil_fun
from Airfoil_simulation_1.ShapeToPerformance import shape_to_performance as STP1
from Airfoil_simulation_2.ShapeToPerformance import shape_to_performance as STP2
from Airfoil_simulation_3.ShapeToPerformance import shape_to_performance as STP3
import argparse
import os,sys

DB2 = np.load(rf"DB2.npy",allow_pickle=True).item()
airfoil_shape = DB2["shapes"]
print(airfoil_shape.shape)
DB2_p = DB2["performances"]
if DB2_p == None:
    performance = STP1(airfoil_shape)
else:
    performance = STP1(airfoil_shape)
print(performance)

np.save("performance.npy" , performance)
print("done!!!")