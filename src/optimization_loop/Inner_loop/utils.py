import torch
import scipy.io as sio
import torch.nn as nn
import numpy as np
from pymoo.core.problem import Problem
from models import UA_surrogate_model, MultiLayerPerceptron_forward , MultiLayerPerceptron_forward_classifier

class BO_surrogate_uncertainty(Problem):
    def __init__(self, n_iter=0):
        super().__init__(n_var=384, n_obj=4, xl=0, xu=1)
        
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

        self.UA_surrogate_model = UA_surrogate_model()
        self.UA_surrogate_model.eval()
        # # if we need outo classifier
        # self.classifier_model = MultiLayerPerceptron_forward_classifier(input_size, hidden_size_mu, self.num_classes)
        # classifier_model_model_name = 'UA_surrogate_weights/Constraint_handlers/constraint_handler_%d.ckpt' % (n_iter)
        # a =  torch.load(classifier_model_model_name, map_location=torch.device('cpu'))
        # self.classifier_model.load_state_dict(torch.load(classifier_model_model_name, map_location=torch.device('cpu')))
        # self.classifier_model.eval()

    def _evaluate(self, design, out, *args, **kwargs):
        net_n = len(self.UA_surrogate_model.cl_forward_mlps)
        designs = torch.tensor(design).float()
        batchsize = designs.shape[0]
        reproduced_Performance_ensemble = torch.empty(net_n, batchsize, self.num_classes)
        reproduced_Performance_mu = torch.empty(batchsize, self.num_classes)
        uncertainty_epistemic = torch.empty((batchsize, self.num_classes))
        out_list = self.UA_surrogate_model(designs)
        reproduced_Performance_ensemble = torch.stack(out_list,dim=0)


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


