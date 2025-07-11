import scipy.io as sio
import numpy as np
# from tf_analytic_airfoil import airfoil_fun
from Airfoil_simulation_1.ShapeToPerformance import shape_to_performance as STP1
from Airfoil_simulation_2.ShapeToPerformance import shape_to_performance as STP2
from Airfoil_simulation_3.ShapeToPerformance import shape_to_performance as STP3
import argparse


airfoil_shape = np.load(rf"generated_airfoils_dpp_0.1_RMS_fixed_auto_norm_false(2).npy")
print(airfoil_shape.shape)
performance = STP1(airfoil_shape[:3,:,:])
print(performance)