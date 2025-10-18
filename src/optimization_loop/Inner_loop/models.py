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

# class MultiLayerPerceptron_forward_classifier(nn.Module):
#     def __init__(self, input_size, hidden_layers, num_classes):
#         super(MultiLayerPerceptron_forward_classifier, self).__init__()
#         #################################################################################
#         # Initialize the modules required to implement the mlp with given layer   #
#         # configuration. input_size --> hidden_layers[0] --> hidden_layers[1] .... -->  #
#         # hidden_layers[-1] --> num_classes                                             #
#         #################################################################################
#         layers = []
#         layers.append(nn.Linear((input_size), (hidden_layers[0])))
#         # layers.append(nn.Linear((hidden_layers[0]), (hidden_layers[1])))
#         # layers.append(nn.Linear((hidden_layers[1]), (hidden_layers[2])))
#         for i in range(len(hidden_layers)-1):
#             layers.append(nn.Linear((hidden_layers[i]), (hidden_layers[i+1])))

#         layers.append(nn.Linear((hidden_layers[len(hidden_layers)-1]), (num_classes)))
#         self.layers = nn.Sequential(*layers)
#         self.hidden_size = hidden_layers
    
#     def forward(self, x):
#         #################################################################################
#         # Implement the forward pass computations                                 #
#         #################################################################################

#         # x = F.relu(self.layers[0](x))
#         # x = F.relu(self.layers[1](x))
#         # x = F.relu(self.layers[2](x))
#         for i in range(len(self.hidden_size)):
#             x = F.relu(self.layers[i](x))
#         x = (self.layers[len(self.hidden_size)](x))
#         out = x
#         # out = F.sigmoid(x)
#         return out


class MultiLayerPerceptron_forward(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes, net_n, dropout_prob=0.2):
        super(MultiLayerPerceptron_forward, self).__init__()

        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()  # BatchNorm layers
        self.dropout = nn.Dropout(p=dropout_prob)
        self.net_n = net_n

        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_layers[0]))
        self.bns.append(nn.BatchNorm1d(hidden_layers[0]))

        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            self.bns.append(nn.BatchNorm1d(hidden_layers[i+1]))

        # Output layer
        self.layers.append(nn.Linear(hidden_layers[-1], num_classes))

    def forward(self, x):
        m = activation_function_list[self.net_n]

        # Forward pass through hidden layers
        for i in range(len(self.bns)):
            x = self.layers[i](x)
            x = self.bns[i](x)
            x = m(x)
            x = self.dropout(x)  # Apply dropout after activation

        # Output layer (no activation)
        x = self.layers[-1](x)
        return x

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
                [150,200,200,150],
                [150,200,200,150],
                [150,200,200,150],
                [150,200,200,150],
                [150,200,200,150]
                ],
                hidden_layers_cd_models = [
                [200,300,300,200],
                [200,300,300,200],
                [200,300,300,200],
                [200,300,300,200],
                [200,300,300,200],
                [200,300,300,200],
                [200,300,300,200],
                [200,300,300,200]
                ], 
                net_n_cl= [0,2,3,4,0,2,3,4], 
                net_n_cd= [0,2,3,4,0,2,3,4], 
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
        self.net_n_cl = net_n_cl
        self.net_n_cd = net_n_cd
        super(UA_surrogate_model,self).__init__()
        self.cl_forward_mlps = nn.ModuleList()
        self.cd_forward_mlps = nn.ModuleList()
        for i in range(len(net_n_cl)):
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
        
        self.init_weigths()
    
    def init_weigths(self):
        """
            initializingthe weigths,
            for relu  types: [1, 2, 3, 4] -> relubased -> kaiming init
            for relu  types: [0] -> relubased -> xaier uniform init
        """

        def init_mlp_weights(mlp, activation_type):
            # Map activation index to type name
            # Adjust based on your activation_function_list
            relu_like = [1, 2, 3, 4]   # ReLU-family activations
            tanh_like = [0]                 # tanh-based activations

            for layer in mlp.modules():
                if isinstance(layer, nn.Linear):
                    if activation_type in relu_like:
                        nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                    elif activation_type in tanh_like:
                        nn.init.xavier_uniform_(layer.weight)
                    else:
                        # fallback to Xavier if unknown
                        nn.init.xavier_uniform_(layer.weight)
                    
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

        # Initialize cl (lift coefficient) networks
        for mlp, net_type in zip(self.cl_forward_mlps, self.net_n_cl):
            init_mlp_weights(mlp, net_type)

        # Initialize cd (drag coefficient) networks
        for mlp, net_type in zip(self.cd_forward_mlps, self.net_n_cd):
            init_mlp_weights(mlp, net_type)

    def forward(self, x , Eps = 1e-10):
        #################################################################################
        # Implement the forward pass computations                                 #
        #################################################################################
        
        each_line = []
        for i in range(len(self.cl_forward_mlps)):
            each_line.append(torch.concat([self.cl_forward_mlps[i](x),self.cl_forward_mlps[i](x) / (self.cd_forward_mlps[i](x) + Eps)],dim=1))
        

        return each_line

    def get_cl_cd(self,x):
        cl = []
        cd = []
        for i in range(len(self.cl_forward_mlps)):
            cl.append(self.cl_forward_mlps[i](x))
            cd.append(self.cd_forward_mlps[i](x))
        cl =torch.stack(cl,dim=0)
        cd = torch.stack(cd,dim=0)
        # print(f"{cl.shape=}")
        # print(f"{cd.shape=}")
        ans = torch.concat([cl,cd],dim=-1)
        # print(f"{ans.shape=}")
        reproduced_Performance_mu = (1 / len(self.cl_forward_mlps)) * torch.sum(ans, dim=0)
        # print(f"{reproduced_Performance_mu.shape=}")
        return reproduced_Performance_mu

class Best_surrogate_model(nn.modules):
    """
        input_shape = flat(batch , 192 , 2) -> (batch , 384)
        output_shape = list of the (batch,2)  shapes for each MLP
    """
    def __init__(self,
                input_size =  192 * 2, 
                hidden_layers = [
                [150,200,200,150],
                [150,200,200,150],
                [150,200,200,150],
                [150,200,200,150],
                [150,200,200,150],
                [150,200,200,150],
                [150,200,200,150],
                [150,200,200,150],
                [150,200,200,150],
                [150,200,200,150]
                ],
                net_n= [0,1,2,3,4,0,1,2,3,4],  
                path_models =None, 

                ):
        self.net_n = net_n
        super(Best_surrogate_model,self).__init__()
        self.forward_mlps = nn.ModuleList()
        for i in range(len(net_n)):
            self.forward_mlps.append(
                MultiLayerPerceptron_forward(input_size , hidden_layers[i] ,   num_classes=2  , net_n=net_n[i])
            )
            if path_models:
                self.forward_mlps[i].load_state_dict(torch.load(path_models[i],map_location="cpu" ,weights_only=True))
        
        self.init_weigths()
    
    def init_weigths(self):
        """
            initializingthe weigths,
            for relu  types: [1, 2, 3, 4] -> relubased -> kaiming init
            for relu  types: [0] -> relubased -> xaier uniform init
        """

        def init_mlp_weights(mlp, activation_type):
            # Map activation index to type name
            # Adjust based on your activation_function_list
            relu_like = [1, 2, 3, 4]   # ReLU-family activations
            tanh_like = [0]                 # tanh-based activations

            for layer in mlp.modules():
                if isinstance(layer, nn.Linear):
                    if activation_type in relu_like:
                        nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                    elif activation_type in tanh_like:
                        nn.init.xavier_uniform_(layer.weight)
                    else:
                        # fallback to Xavier if unknown
                        nn.init.xavier_uniform_(layer.weight)
                    
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

        # Initialize cl (lift coefficient) networks
        for mlp, net_type in zip(self.forward_mlps, self.net_n):
            init_mlp_weights(mlp, net_type)


    def forward(self, x , Eps = 1e-10):
        #################################################################################
        # Implement the forward pass computations                                 #
        #################################################################################
        
        each_line = []
        for i in range(len(self.forward_mlps)):
            cl , cd = self.forward_mlps[i](x)
            each_line.append(torch.concat([cl,cl / (cd + Eps)],dim=1))
        

        return each_line

    def get_cl_cd(self,x):
        cl = []
        cd = []
        for i in range(len(self.cl_forward_mlps)):
            cl_temp , cd_temp = self.forward_mlps[i](x)
            cl.append(cl_temp)
            cd.append(cd_temp)

        cl =torch.stack(cl,dim=0)
        cd = torch.stack(cd,dim=0)
        # print(f"{cl.shape=}")
        # print(f"{cd.shape=}")
        ans = torch.concat([cl,cd],dim=-1)
        # print(f"{ans.shape=}")
        reproduced_Performance_mu = (1 / len(self.forward_mlps)) * torch.sum(ans, dim=0)
        # print(f"{reproduced_Performance_mu.shape=}")
        return reproduced_Performance_mu


    
if __name__  == "__main__":
    model = UA_surrogate_model()
    x = torch.zeros((2,384))
    out = model.get_cl_cd(x)
    print(out)
