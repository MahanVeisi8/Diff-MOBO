import scipy.io as sio
import numpy as np
# from tf_analytic_airfoil import airfoil_fun
from Airfoil_simulation_1.ShapeToPerformance import shape_to_performance as STP1
from Airfoil_simulation_2.ShapeToPerformance import shape_to_performance as STP2
from Airfoil_simulation_3.ShapeToPerformance import shape_to_performance as STP3
import argparse
import os,sys
import argparse


NUM_CORES = 200


Db = np.load(rf"run_results_xstrain_final.npy",allow_pickle=True).item()
xs_train = np.load(rf"xs_train.npy",allow_pickle=True)
# s = Db["shapes"]
# l = Db["latents"]
# p = Db["performances"]
# for sample in  p:
#     print(sample)
# print(f"{s.shape}")
# print(f"{l.shape}") 
# print(f"{p.shape}")
xs_train = xs_train[:3]

file_shapes = Db["shapes"]
file_performances = Db["performances"]
print(f"for the first {len(xs_train)} in the  xs_train.npy")
l = np.zeros((len(xs_train),2,192))
p , s , l = STP1(xs_train ,l)
print(p)
print(f"finding in the indices")
for s in xs_train:
    for i in range(len(file_shapes)):
        shape = file_shapes[i]
        performance = file_performances[i]
        if np.array_equal(shape, s):
            print(performance)
# print("in the file")
# print(p)
# l = Db["latents"][:3,:,:]
# p , s , l = STP1(shapes ,l)
# print("in the system")
# print(p)
# print(np.equal(shapes, s))
# indices = np.argsort(p[:, 2], axis=0)
# print(f"{np.unique(p[:,2])}")
# f = p[indices]
# for i  in range (5):
#     print(f[i])
# shapes = Db["shapes"][[0,49,99,149,199]]
# p = Db["performances"][[0,49,99,149,199]]
# l = Db["latents"][[0,49,99,149,199]]
# performance = STP1(shapes ,l)
# print(f"{shapes.shape}")
# print(p)
# print(performance[0])
# p = Db["performances"][:2,:]
# print(performance[0])