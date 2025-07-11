from tf_analytic_airfoil import airfoil_fun
from scipy.io import loadmat
from scipy.io import savemat
import numpy as np


    
    
model = airfoil_fun()
design = loadmat('Dataset/GAN4/gan4_latent.mat')
design = design['gan4_latent']



synthesized_airfoil = model.airfoil_analytic(design, return_values_of=['airfoil'])
synthesized_airfoil = np.squeeze(synthesized_airfoil)
# savemat('./Dataset/GAN2_data_shapes.mat', {'shapes': np.array(synthesized_airfoil)})
savemat('Dataset/GAN4/GAN4_shapes.mat', {'shapes': np.array(synthesized_airfoil)})