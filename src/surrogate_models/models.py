import sys, os
import torch.nn as nn
import torch.nn.functional as F
import torch

activation_function_list = [torch.tanh, nn.ReLU(), nn.CELU(), nn.LeakyReLU(), nn.ELU(), nn.Hardswish(),torch.tanh, nn.ReLU(), nn.CELU(), nn.LeakyReLU(), torch.tanh]

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

#===================================================    

import torch
import torch.nn as nn
import torch.nn.functional as F

# List of activation functions
activation_function_list = [
    torch.tanh, nn.ReLU(), nn.CELU(), nn.LeakyReLU(), nn.ELU(),
    nn.Hardswish(), torch.tanh, nn.ReLU(), nn.CELU(), nn.LeakyReLU(), torch.tanh
]

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim

        # Learnable linear projections
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key   = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.scale = embed_dim ** 0.5

    def forward(self, x):
        # x shape: (batch, seq_len, embed_dim)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Attention scores: (batch, seq_len, seq_len)
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        attn_weights = F.softmax(scores, dim=-1)

        # Weighted sum: (batch, seq_len, embed_dim)
        attn_output = torch.bmm(attn_weights, V)
        return attn_output


class MultiLayerPerceptronWithCustomAttention(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes, net_n):
        super(MultiLayerPerceptronWithCustomAttention, self).__init__()

        # MLP Layers
        layers = []
        layers.append(nn.Linear(input_size, hidden_layers[0]))
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
        self.mlp_layers = nn.ModuleList(layers)
        self.activation = activation_function_list[net_n]
        
        self.attention = SelfAttention(embed_dim=hidden_layers[-1])
        self.final_linear = nn.Linear(hidden_layers[-1], num_classes)

    def forward(self, x):
        # MLP forward
        for layer in self.mlp_layers:
            x = self.activation(layer(x))

        # Reshape for attention: treat features as a sequence of length 1
        x = x.unsqueeze(1)  # (batch, 1, embed_dim)

        # Apply attention
        x = self.attention(x)

        # Flatten back to (batch, embed_dim)
        x = x.squeeze(1)

        # Final output
        out = self.final_linear(x)
        return out

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

#====================================================

class ResidualBlock1D(nn.Module):
    """
    A 1D residual block with optional downsampling.
    The first conv can do stride=2 if downsample=True, the second conv uses stride=1.
    We'll use kernel=5, pad=2 so that stride=1 doesn't reduce length at all.
    """
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        self.downsample = downsample

        # If we want to downsample, we set stride=2 for the first conv and the shortcut
        stride_first = 2 if downsample else 1
        kernel_size = 5
        pad = 2

        # 1) First conv
        self.bn1 = nn.BatchNorm1d(in_channels, momentum=0.9)
        self.conv1 = nn.Conv1d(in_channels, out_channels,
                               kernel_size=kernel_size,
                               stride=stride_first,
                               padding=pad)
        # 2) Second conv always stride=1
        self.bn2 = nn.BatchNorm1d(out_channels, momentum=0.9)
        self.conv2 = nn.Conv1d(out_channels, out_channels,
                               kernel_size=kernel_size,
                               stride=1,
                               padding=pad)

        # Shortcut (for downsample or channel mismatch)
        if downsample or (in_channels != out_channels):
            self.shortcut_conv = nn.Conv1d(in_channels, out_channels,
                                           kernel_size=1,
                                           stride=stride_first,
                                           padding=0)
        else:
            self.shortcut_conv = nn.Identity()

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, debug=False):
        if debug:
            print(f"  [ResidualBlock] Input shape: {x.shape}")
        identity = self.shortcut_conv(x)     # Possibly stride=2 if downsample is True

        # First conv
        out = self.bn1(x)
        out = self.lrelu(out)
        out = self.conv1(out)
        if debug:
            print(f"    After conv1 shape: {out.shape}")

        # Second conv
        out = self.bn2(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        if debug:
            print(f"    After conv2 shape: {out.shape}")

        # Add skip
        out = out + identity
        if debug:
            print(f"  [ResidualBlock] Output shape: {out.shape}\n")

        return out


class SurrogateModel1D(nn.Module):
    """
    A 1D ResNet-like approach for (batch, 2, n_points).
    We'll do four "groups" of blocks, each of which can downsample the length dimension in the first block.
    """
    def __init__(self, n_points=192, in_channels=2, depth=16, residual_list=[2,2,2,2]):
        super().__init__()
        self.depth = depth
        self.residual_list = residual_list

        # -- Initial Conv: kernel=5, pad=2, stride=1 => length stays 192
        self.conv_initial = nn.Conv1d(in_channels, depth, kernel_size=5, stride=1, padding=2)
        self.bn_initial = nn.BatchNorm1d(depth, momentum=0.9)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        # Group0: no downsample in first block
        self.blocks0 = nn.ModuleList([
            ResidualBlock1D(depth, depth, downsample=False)
            for _ in range(residual_list[0])
        ])
        # Group1: first block downsample => depth-> depth*2
        self.resblock1_0 = ResidualBlock1D(depth, depth*2, downsample=True)
        self.blocks1 = nn.ModuleList([
            ResidualBlock1D(depth*2, depth*2, downsample=False)
            for _ in range(residual_list[1]-1)
        ])
        # Group2
        self.resblock2_0 = ResidualBlock1D(depth*2, depth*4, downsample=True)
        self.blocks2 = nn.ModuleList([
            ResidualBlock1D(depth*4, depth*4, downsample=False)
            for _ in range(residual_list[2]-1)
        ])
        # Group3
        self.resblock3_0 = ResidualBlock1D(depth*4, depth*8, downsample=True)
        self.blocks3 = nn.ModuleList([
            ResidualBlock1D(depth*8, depth*8, downsample=False)
            for _ in range(residual_list[3]-1)
        ])

        # Final BN + global avg pool + FC(128) + BN + LReLU + FC(2) + Sigmoid
        self.bn_final = nn.BatchNorm1d(depth*8, momentum=0.9)
        self.fc1 = nn.Linear(depth*8, 128)
        self.bn_fc1 = nn.BatchNorm1d(128, momentum=0.9)
        self.fc2 = nn.Linear(128, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, debug=False):
        if debug:
            print(f"Input shape: {x.shape}")
        out = self.conv_initial(x)
        out = self.bn_initial(out)
        out = self.lrelu(out)
        if debug:
            print(f"After initial conv shape: {out.shape}")

        # group0
        for block in self.blocks0:
            out = block(out, debug=debug)
        if debug:
            print(f"After group0 shape: {out.shape}")

        # group1
        out = self.resblock1_0(out, debug=debug)
        for block in self.blocks1:
            out = block(out, debug=debug)
        if debug:
            print(f"After group1 shape: {out.shape}")

        # group2
        out = self.resblock2_0(out, debug=debug)
        for block in self.blocks2:
            out = block(out, debug=debug)
        if debug:
            print(f"After group2 shape: {out.shape}")

        # group3
        out = self.resblock3_0(out, debug=debug)
        for block in self.blocks3:
            out = block(out, debug=debug)
        if debug:
            print(f"After group3 shape: {out.shape}")

        out = self.bn_final(out)
        out = self.lrelu(out)
        if debug:
            print(f"After bn_final + lrelu shape: {out.shape}")

        # global avg pool => (batch, depth*8, 1)
        out = F.adaptive_avg_pool1d(out, 1)
        if debug:
            print(f"After global avg pool shape: {out.shape}")
        out = out.view(out.size(0), -1)  # => (batch, depth*8)

        out = self.fc1(out)
        out = self.bn_fc1(out)
        out = self.lrelu(out)
        if debug:
            print(f"After fc1 shape: {out.shape}")

        out = self.fc2(out)
        out = self.sigmoid(out)
        if debug:
            print(f"Output shape: {out.shape}")

        return out

#====================================================

class Hybrid_surrogate_MLP(nn.Module):
    def __init__(self, 
                 input_size, 
                 hidden_layers_cl_model ,
                 hidden_layers_cd_model , 
                 net_n_cl=3 , 
                 net_n_cd=3, 
                 path_cl_model = None, 
                 path_cd_model  =  None):
        super(Hybrid_surrogate_MLP, self).__init__()
        self.cl_forward_mlp = MultiLayerPerceptron_forward(input_size , hidden_layers_cl_model ,   num_classes=1  , net_n=net_n_cl)
        self.cd_forward_mlp = MultiLayerPerceptron_forward(input_size , hidden_layers_cd_model ,   num_classes=1  , net_n=net_n_cd)
        if path_cl_model:
            self.cl_forward_mlp.load_state_dict(torch.load(path_cl_model,map_location="cpu"))
        if path_cd_model:
            self.cd_forward_mlp.load_state_dict(torch.load(path_cd_model,map_location="cpu"))

    def forward(self, x):
        #################################################################################
        # Implement the forward pass computations                                 #
        #################################################################################
        cl = self.cl_forward_mlp(x)
        cd = self.cd_forward_mlp(x)
        return torch.stack([cl,cd],dim=1).squeeze(-1)
 

