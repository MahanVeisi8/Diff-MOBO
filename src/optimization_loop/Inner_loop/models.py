import torch
import scipy.io as sio
import torch.nn as nn
import numpy as np
from pymoo.core.problem import Problem
import os,sys
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")

activation_function_list = [torch.tanh, nn.ReLU(), nn.CELU(), nn.LeakyReLU(), nn.ELU(), nn.Hardswish(),torch.tanh, nn.ReLU(), nn.CELU(), nn.LeakyReLU(), torch.tanh]

class MultiLayerPerceptron_forward_classifier(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes):
        super(MultiLayerPerceptron_forward_classifier, self).__init__()
        #################################################################################
        # Initialize the modules required to implement the mlp with given layer   #
        # configuration. input_size --> hidden_layers[0] --> hidden_layers[1] .... -->  #
        # hidden_layers[-1] --> num_classes                                             #
        #################################################################################
        layers = []
        layers.append(nn.Linear((input_size), (hidden_layers[0])))
        # layers.append(nn.Linear((hidden_layers[0]), (hidden_layers[1])))
        # layers.append(nn.Linear((hidden_layers[1]), (hidden_layers[2])))
        for i in range(len(hidden_layers)-1):
            layers.append(nn.Linear((hidden_layers[i]), (hidden_layers[i+1])))

        layers.append(nn.Linear((hidden_layers[len(hidden_layers)-1]), (num_classes)))
        self.layers = nn.Sequential(*layers)
        self.hidden_size = hidden_layers
    
    def forward(self, x):
        #################################################################################
        # Implement the forward pass computations                                 #
        #################################################################################

        # x = F.relu(self.layers[0](x))
        # x = F.relu(self.layers[1](x))
        # x = F.relu(self.layers[2](x))
        for i in range(len(self.hidden_size)):
            x = F.relu(self.layers[i](x))
        x = (self.layers[len(self.hidden_size)](x))
        out = x
        # out = F.sigmoid(x)
        return out

class MultiLayerPerceptron_forward(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes, net_n):
        super(MultiLayerPerceptron_forward, self).__init__()
        #################################################################################
        # Initialize the modules required to implement the mlp with given layer   #
        # configuration. input_size --> hidden_layers[0] --> hidden_layers[1] .... -->  #
        # hidden_layers[-1] --> num_classes                                             #
        #################################################################################
        layers = []
        layers.append(nn.Linear((input_size), (hidden_layers[0])))
        for i in range(len(hidden_layers)-1):
            layers.append(nn.Linear((hidden_layers[i]), (hidden_layers[i+1])))

        layers.append(nn.Linear((hidden_layers[len(hidden_layers)-1]), (num_classes)))
        self.layers = nn.Sequential(*layers)
        self.net_n = net_n
        self.hidden_layers = hidden_layers
    def forward(self, x):
        #################################################################################
        # Implement the forward pass computations                                 #
        #################################################################################
        m = activation_function_list[self.net_n]
        for i in range(len(self.hidden_layers)):
            x = self.layers[i](x)
            x = m(x)
        x = (self.layers[len(self.hidden_layers)](x))
        out=x
        return out

class UA_surrogate_model(nn.Module):
    """
        input_shape = flat(batch , 192 , 2) -> (batch , 384)
        output_shape = list of the (batch,2)  shapes for each MLP
    """
    def __init__(self,
                input_size =  192 * 2, 
                hidden_layers_cl_models = [
                [150,200,200,150],
                [150,200,200,150],
                [150,200,200,150],
                [100,300,300,100],
                ],
                hidden_layers_cd_models = [
                [200,300,300,200],
                [200,300,300,200],
                [200,300,300,200],
                [200,300,300,200],
                ], 
                net_n_cl= [0,2,3,4], 
                net_n_cd= [0,2,3,4], 
                path_cl_models =None, 
                path_cd_models =None
                # path_cl_models  = [
                # rf"UA_surrogate_weights/cl/0_[150-200-200-150]/mlp_best_model.pt",
                # rf"UA_surrogate_weights/cl/2_[150-200-200-150]/mlp_best_model.pt",
                # rf"UA_surrogate_weights/cl/3_[150-200-200-150]/mlp_best_model.pt",
                # rf"UA_surrogate_weights/cl/4_[100-300-300-100]/mlp_best_model.pt"
                # ], 
                # path_cd_models = [
                #     rf"UA_surrogate_weights/cd/0_[200-300-300-200]/mlp_best_model.pt",
                #     rf"UA_surrogate_weights/cd/2_[200-300-300-200]/mlp_best_model.pt",
                #     rf"UA_surrogate_weights/cd/3_[200-300-300-200]/mlp_best_model.pt",
                #     rf"UA_surrogate_weights/cd/4_[200-300-300-200]/mlp_best_model.pt"
                # ],
                ):
        super(UA_surrogate_model,self).__init__()
        self.cl_forward_mlps = nn.ModuleList()
        self.cd_forward_mlps = nn.ModuleList()
        for i in range(4):
            self.cl_forward_mlps.append(
                MultiLayerPerceptron_forward(input_size , hidden_layers_cl_models[i] ,   num_classes=1  , net_n=net_n_cl[i])
            )
            self.cd_forward_mlps.append(
                MultiLayerPerceptron_forward(input_size , hidden_layers_cd_models[i] ,   num_classes=1  , net_n=net_n_cd[i])
            )
            if path_cl_models:
                self.cl_forward_mlps[i].load_state_dict(torch.load(path_cl_models[i],map_location="cpu" ,weights_only=True))
            if path_cd_models:
                self.cd_forward_mlps[i].load_state_dict(torch.load(path_cd_models[i],map_location="cpu" ,weights_only=True))
    
    def forward(self, x , Eps = 1e-10):
        #################################################################################
        # Implement the forward pass computations                                 #
        #################################################################################
        
        each_line = []
        for i in range(len(self.cl_forward_mlps)):
            each_line.append(torch.concat([self.cl_forward_mlps[i](x),self.cl_forward_mlps[i](x) / (self.cd_forward_mlps[i](x) + Eps)],dim=1))
        

        return each_line


if __name__  == "__main__":
    model = UA_surrogate_model()
    x = torch.zeros((2,384))
    out = model(x)
    print(out)
