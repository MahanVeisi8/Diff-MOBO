import torch
import scipy.io as sio
import torch.nn as nn
import numpy as np
from pymoo.core.problem import Problem
from models import UA_surrogate_model, MultiLayerPerceptron_forward , MultiLayerPerceptron_forward_classifier
from pathlib import Path
import torch
from torch.utils.data import TensorDataset, DataLoader

DATA_DIR = Path(rf"../../../data")
coord_mm = np.load(DATA_DIR/"coord_min_max.npy")  # [[x_min,y_min],[x_max,y_max]]
x_min,y_min = coord_mm[0]; x_max,y_max = coord_mm[1]

class BO_surrogate_uncertainty(Problem):
    def __init__(self,diffusion,num_cores=2, device = "cuda" ,n_iter=0 ):
        super().__init__(n_var=384, n_obj=4, xl=0, xu=1)
        
        # setting up the diffusion models for getting designs out of latents
        self.diffusion = diffusion
        self.device = device
        self.num_cores= num_cores
        # Load the forward model    
        input_size = 384
        hidden_size_mu = [150, 200, 200 , 150]
        self.num_classes = 2
        
        # Load the models
        # self.mu_models = []
        # for n_net in range(1, 11):
        #     # Load mu models
        #     mu_model = MultiLayerPerceptron_forward(input_size, hidden_size_mu, self.num_classes, n_net)
        #     model_name = 'Models/iter_%d/mu_net_%d.ckpt' % (n_iter, n_net)
        #     mu_model.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))
        #     mu_model.eval()
        #     self.mu_models.append(mu_model)

        self.UA_surrogate_model = UA_surrogate_model().to(self.device)
        self.UA_surrogate_model.eval()
        # # if we need outo classifier
        # self.classifier_model = MultiLayerPerceptron_forward_classifier(input_size, hidden_size_mu, self.num_classes)
        # classifier_model_model_name = 'UA_surrogate_weights/Constraint_handlers/constraint_handler_%d.ckpt' % (n_iter)
        # a =  torch.load(classifier_model_model_name, map_location=torch.device('cpu'))
        # self.classifier_model.load_state_dict(torch.load(classifier_model_model_name, map_location=torch.device('cpu')))
        # self.classifier_model.eval()

    def _evaluate(self, latents, out, *args, **kwargs):
        net_n = len(self.UA_surrogate_model.cl_forward_mlps)
        # getting airfoil designs out of the latents
        latents = torch.from_numpy(latents).float() # this latent has  been generated form the algorithm (batch , 384)
        designs = self._latents_to_shapes(latents.reshape(latents.shape[0] , 2,  -1))

        # getting the UA infromations from the  airfoil designs 
        batchsize = designs.shape[0]
        reproduced_Performance_ensemble = torch.empty(net_n, batchsize, self.num_classes)
        reproduced_Performance_mu = torch.empty(batchsize, self.num_classes)
        uncertainty_epistemic = torch.empty((batchsize, self.num_classes))

        out_list = self.UA_surrogate_model(designs.reshape(batchsize, -1).to(self.device))
        reproduced_Performance_ensemble = torch.stack(out_list,dim=0).to("cpu")

        # calculating the mean  and  epistemic variance
        reproduced_Performance_mu = (1 / 10) * torch.sum(reproduced_Performance_ensemble, 0)
        uncertainty_epistemic = (1 / 10) * torch.sum(
            reproduced_Performance_ensemble ** 2 - reproduced_Performance_mu.repeat(net_n, 1, 1) ** 2,
            0)
        
        # # if we need outo classifier
        # out_classifier = self.classifier_model(designs)
        # validity_score = ((out_classifier[:, 0:1].repeat(1, 2).detach()) - (out_classifier[:, 1:2].repeat(1, 2).detach())) / 200
        validity_score = 0

        out["F"] = torch.cat((-reproduced_Performance_mu[:, :2].detach() + validity_score, -uncertainty_epistemic[:, :2] + validity_score/10), 1)
        out["F"] = out["F"].detach().numpy()

    def _latents_to_shapes(self, latents , BATCH_SIZE = 128):
        """
        Converts latents (batch, 2, 192) to generated shapes (batch, 192, 2)
        using self.diffusion.latent_sample and inv_coords.
        Uses DataLoader for CPU batching and optional parallelism.
        """
        if isinstance(latents, np.ndarray):
            latents = torch.from_numpy(latents).float().to(self.device)
        device = self.device
        num_to_generate = len(latents)
        batch_size      = BATCH_SIZE

        dataset = TensorDataset(latents)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=self.num_cores-1)

        all_shapes = []
        for i, batch in enumerate(loader):
            latent_batch = batch[0].to(self.device)  # shape: (B, 2, 192)
            with torch.no_grad():
                samples = self.diffusion.latent_sample(latent_batch, is_ddim=True)
                generated_real = self.inv_coords(samples)
                all_shapes.append(generated_real)
            print(f"Processed { (i+1)*batch_size } / { len(latents) } latents")

        # Stack all outputs
        return torch.from_numpy(np.vstack(all_shapes)).to("cpu")

    def inv_coords(self, xs_s):                     # xs_s shape (...,2,192) tensor
        xs_np = xs_s.permute(0,2,1).cpu().numpy()   # -> (B,192,2)
        xs_np[...,0] = xs_np[...,0]*(x_max-x_min) + x_min
        xs_np[...,1] = xs_np[...,1]*(y_max-y_min) + y_min
        return xs_np                                # (B,192,2) numpy
    