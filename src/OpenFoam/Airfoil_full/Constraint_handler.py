import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset
import scipy.io as sio
import argparse
import numpy as np
import time
from layer_config_forward_classifier import MultiLayerPerceptron_forward_classifier


parser = argparse.ArgumentParser()
parser.add_argument("-iter_num", type=int,
                    help="iteration_number")
args = parser.parse_args()
iter_n = args.iter_num

def weights_init(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0.0, 1e-3)
        m.bias.data.fill_(0.)

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
#--------------------------------
# Device configuration
#--------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s'%device)

#--------------------------------
# Hyper-parameters
#--------------------------------
input_size = 5
hidden_size = [150, 200, 200 , 150]
num_classes = 2
num_epochs = 150
# batch_size = 200
learning_rate = 9*1e-4
# learning_rate = 0.0001
learning_rate_decay = 0.98
reg = 0.001
batch_size = 100


# Load the data set
Design_data_all = np.array([])
Performance_data_all = np.array([])
for dset in range(iter_n+1):
    mat = sio.loadmat('Dataset/latent_%d.mat' %(dset))
    mat = mat['dset']
    Design_data = mat['latent_all'][0][0]
    Performance_data = mat['vaiolation_flag'][0][0]
    if dset == 0:  # Initialize arrays in the first iteration
        Design_data_all = Design_data
        Performance_data_all = Performance_data
    else:
        Design_data_all = np.concatenate((Design_data_all, Design_data), axis=0)
        Performance_data_all = np.concatenate((Performance_data_all, Performance_data), axis=1)

Design_data = Design_data_all
Performance_data = Performance_data_all
Prformance_data = Performance_data.squeeze()





print(Prformance_data.shape)
print(Design_data.shape)
x_train_tensor = torch.from_numpy(Design_data).float()
y_train_tensor = torch.from_numpy(Prformance_data).float()


dataset = TensorDataset(x_train_tensor, y_train_tensor)

lengths = [int(len(dataset)*0.9), len(dataset)-int(len(dataset)*0.9)]
train_dataset, val_dataset = torch.utils.data.random_split(dataset, lengths)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size = lengths[1])



model_forward = MultiLayerPerceptron_forward_classifier(input_size, hidden_size, num_classes).to(device)
print(count_parameters(model_forward))

model_forward.apply(weights_init)
model_forward.to(device)


# criterion = nn.BCELoss()
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model_forward.parameters(), lr=learning_rate, weight_decay=0.0001 )

# Train the model_forward
lr = learning_rate
total_step = len(train_loader)
time_start = time.time()
for epoch in range(num_epochs):
    for i, (controll, state) in enumerate(train_loader):
        # Move tensors to the configured device
        controll = controll.to(device)
        state = state.to(device)
        #################################################################################
        # Implement the training code                                             #
        optimizer.zero_grad()
        # im = controll.view(31, input_size)
        outputs = model_forward(controll)
        loss = criterion(outputs, state.long())
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    # Code to update the lr
    lr *= learning_rate_decay
    update_lr(optimizer, lr)
    with torch.no_grad():
        correct = 0
        total = 0
        for controll, state in val_loader:
            state = state.to(device)
            ####################################################
            #evaluation #
            controll = controll.to(device)
            # outputs = model_forward(controll.view(31, input_size))
            outputs = model_forward(controll)
            loss = criterion(outputs, state.long())
            # loss = ((lab_color_gt - lab_color_output)**2).mean(axis=None)
            print('Validataion MSE is: {}'.format(loss))
            one_epoch_time = time.time() - time_start
            # print(one_epoch_time)
            _, predicted = torch.max(outputs.data, 1)
            total += state.size(0)
            correct += (predicted == state).sum().item()
            print(f'Accuracy of the network on test : {100 * correct / total} %')




# save the model
model_name = 'Models/iter_%d/constraint_handler.ckpt' %(iter_n)
torch.save(model_forward.state_dict(), model_name)




