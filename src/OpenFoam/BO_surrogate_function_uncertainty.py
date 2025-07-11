import torch
import scipy.io as sio
import torch.nn as nn
import numpy as np
from pymoo.core.problem import Problem
from layer_config_forward import MultiLayerPerceptron_forward
from layer_config_forward_classifier import MultiLayerPerceptron_forward_classifier


class BO_surrogate_uncertainty(Problem):
    def __init__(self, n_iter):
        super().__init__(n_var=5, n_obj=4, xl=0, xu=1)
        
        # Load the forward model
        input_size = 5
        hidden_size_mu = [150, 200, 200 , 150]
        self.num_classes = 2
        
        # Load the models
        self.mu_models = []
        for n_net in range(1, 11):
            # Load mu models
            mu_model = MultiLayerPerceptron_forward(input_size, hidden_size_mu, self.num_classes, n_net)
            model_name = 'Models/iter_%d/mu_net_%d.ckpt' % (n_iter, n_net)
            mu_model.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))
            mu_model.eval()
            self.mu_models.append(mu_model)

        self.classifier_model = MultiLayerPerceptron_forward_classifier(input_size, hidden_size_mu, self.num_classes)
        classifier_model_model_name = 'Models/iter_%d/constraint_handler.ckpt' % (n_iter)
        self.classifier_model.load_state_dict(torch.load(classifier_model_model_name, map_location=torch.device('cpu')))
        self.classifier_model.eval()
        
    def _evaluate(self, design, out, *args, **kwargs):
        designs = torch.tensor(design).float()
        batchsize = designs.shape[0]
        reproduced_Performance_ensemble = torch.empty(10, batchsize, self.num_classes)

        reproduced_Performance_mu = torch.empty(batchsize, self.num_classes)
        uncertainty_epistemic = torch.empty((batchsize, self.num_classes))
        for net_n in range(10):
            reproduced_Performance_ensemble[net_n, :, :] = self.mu_models[net_n](designs)

        reproduced_Performance_mu = (1 / 10) * torch.sum(reproduced_Performance_ensemble, 0)
        uncertainty_epistemic = (1 / 10) * torch.sum(
            reproduced_Performance_ensemble ** 2 - reproduced_Performance_mu.repeat(10, 1, 1) ** 2,
            0)

        out_classifier = self.classifier_model(designs)
        validity_score = ((out_classifier[:, 0:1].repeat(1, 2).detach()) - (out_classifier[:, 1:2].repeat(1, 2).detach())) / 200

        out["F"] = torch.cat((-reproduced_Performance_mu[:, :2].detach() + validity_score, -uncertainty_epistemic[:, :2] + validity_score/10), 1)
        out["F"] = out["F"].detach().numpy()

# import torch
# import scipy.io as sio
# import torch.nn as nn
# import numpy as np
# from pymoo.core.problem import Problem
# from layer_config_forward import MultiLayerPerceptron_forward
# from layer_config_forward_classifier import MultiLayerPerceptron_forward_classifier


# class BO_surrogate_uncertainty(Problem):
    # def __init__(self, n_iter):
        # super().__init__(n_var=5,
                         # n_obj=4,
                         # xl=0,
                         # xu=1)

        
        # # Load the forward model
        # input_size = 5
        # hidden_size_mu = [150, 200, 200 , 150]
        # self.num_classes = 2
        # # Load the model
        # self.mu_models = []
        # for n_net in range(1, 11):
            # # load mu models
            # mu_model = MultiLayerPerceptron_forward(input_size, hidden_size_mu, self.num_classes, n_net)
            # model_name = 'Models/iter_%d/mu_net_%d.ckpt' % (n_iter, n_net)
            # mu_model.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))
            # mu_model.eval()
            # self.mu_models.append(mu_model)

        # self.classifier_model = MultiLayerPerceptron_forward_classifier(input_size, hidden_size_mu, self.num_classes)
        # classifier_model_model_name = 'Models/constraint_handler.ckpt'
        # self.classifier_model.load_state_dict(torch.load(classifier_model_model_name, map_location=torch.device('cpu')))
        # self.classifier_model.eval()
        
    # def _evaluate(self, design, out, *args, **kwargs):
        # designs = torch.tensor(design).float()
        # batchsize = designs.shape[0]
        # reproduced_Performance_ensembel = torch.empty(10, batchsize, self.num_classes)

        # reproduced_Performance_mu = torch.empty(batchsize, self.num_classes)
        # uncertainty_epistemic = torch.empty((batchsize, self.num_classes))
        # for net_n in range(10):
            # reproduced_Performance_ensembel[net_n, :, :] = self.mu_models[net_n](designs)

        # reproduced_Performance_mu = (1 / 10) * torch.sum(reproduced_Performance_ensembel, 0)
        # uncertainty_epistemic = (1 / 10) * torch.sum(
            # reproduced_Performance_ensembel ** 2 - reproduced_Performance_mu.repeat(10, 1, 1) ** 2,
            # 0)

        # validity_score = ((out_classifier[:,0:1].repeat(1, 2).detach()) - (out_classifier[:,1:2].repeat(1, 2).detach()))/50

        # out["F"] = torch.cat((-reproduced_Performance_mu[:, :2].detach() + validity_score, -uncertainty_epistemic[:, :2]), 1)
        # out["F"] = out["F"].detach().numpy()
