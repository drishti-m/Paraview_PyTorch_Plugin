# working binary classifier for velocity data
# ref: https://medium.com/@prudhvirajnitjsr/simple-classifier-using-pytorch-37fba175c25c
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math


class Dataset:
    def __init__(self):
        self.input_data = []
        self.output_data = []

    def _addData(self, elements):
        elements = elements[len(elements) % 3:]
        vectors = elements.reshape(-1, 3)
        self.input_data.append(vectors)
        return self.input_data

    def _output_array(self):
        for i_array in self.input_data:
            mag = np.array([np.sqrt(x.dot(x)) for x in i_array])
            output = mag > 0.01000
            output = output.astype(float)

            output = torch.from_numpy(output).to(torch.long)
            self.output_data.append(output)
            # print(output.shape)
        return self.output_data

    def _convert_to_torch(self):
        for idx in range(len(self.input_data)):
            self.input_data[idx] = torch.from_numpy(
                self.input_data[idx]).to(torch.float)

        return self.input_data


def read_arrays(path, delimiter=','):
    arrays_19 = np.genfromtxt(path, delimiter=delimiter)
    arrays_19 = arrays_19[1:]
    arrays_19 = arrays_19[:, 4:7]
    return arrays_19


all_arrays = []
for i in range(0, 13, 1):
    path = 'velocity_data_' + str(i) + ".csv"
    print(path)
    current_array = read_arrays(path)
    all_arrays.append(current_array)
all_arrays = np.array(all_arrays)


dataset = Dataset()

dataset._addData(all_arrays[0])  # , 0:2000])
dataset._addData(all_arrays[1])  # , 500:])
dataset._addData(all_arrays[2])
# dataset._addData(all_arrays[3, 1000:])
dataset._addData(all_arrays[4, 1000:])
# dataset._addData(all_arrays[5, 2000:])
# dataset._addData(all_arrays[6, 200:2000])
dataset._addData(all_arrays[7])  # , 1000:])
dataset._addData(all_arrays[8])  # , 10:2100 ])
dataset._addData(all_arrays[9])
# dataset._addData(all_arrays[10, 1000:2000])
dataset._addData(all_arrays[11])  # , 0:2000])
dataset._addData(all_arrays[12])

dataset._addData(np.arange(-0.1026, -0.00924,  0.0051))
dataset._addData(np.arange(-2.5103, 2.5123, 0.000456))
dataset._addData(np.arange(-8.5134, 8.51234, 0.0001109))
dataset._addData(np.arange(-1.1344, 10.51234, 0.0910))
# dataset._addData(np.arange(0.5000, 12.51234, 0.001234))
# dataset._addData(np.arange(-20.50, 20.51234, 0.0011113))
# dataset._addData(np.arange(-20.51, 20.51234, 0.0004))
# dataset._addData(np.arange(-80.5134, 80.51234, 0.1))
# dataset._addData(np.arange(-10.134, 10.51234, 0.09))
# dataset._addData(np.arange(0.5, 30.5, 0.00001))
# dataset._addData(np.arange(-0.5, 30.5, 0.001))
# dataset._addData(np.arange(1.556, 45.67899, 0.01))
# dataset._addData(np.arange(10.0, 145.67899, 0.134))
# # dataset._addData(np.arange(100.1345, 300.9876, 1.567))
# dataset._addData(np.arange(0.000, 23.9560, 0.00123))
# dataset._addData(np.arange(-3.0345, -1.0345, 0.0000134 ))
input_images = dataset.input_data
output_images = dataset._output_array()
input_images = dataset._convert_to_torch()

validation_dataset = Dataset()
# validation_dataset._addData(all_arrays[0,2000:])
# validation_dataset._addData(all_arrays[1, 0:500])
validation_dataset._addData(all_arrays[3])  # , 0:1000])
# validation_dataset._addData(all_arrays[4, 0:1000])
validation_dataset._addData(all_arrays[5])  # , 0:2000])
validation_dataset._addData(all_arrays[6])  # , 0:200])
# validation_dataset._addData(all_arrays[6, 2000:])
# validation_dataset._addData(all_arrays[7, 0:1000])
validation_dataset._addData(all_arrays[10])  # , 0:500])
# validation_dataset._addData(all_arrays[11, 2000:])
validation_dataset._addData(np.arange(-0.98745, 2.3456, 0.0293))
validation_dataset._addData(np.arange(-0.198, 20.3544, 0.953))
# validation_dataset._addData(np.arange(98.745,234.56, 29.3))
# validation_dataset._addData(np.arange(-0.45,4.56, 0.395))
# validation_dataset._addData(np.arange(-0.87156,-0.005209, 0.01293))

validation_input = validation_dataset.input_data
validation_output = validation_dataset._output_array()
validation_input = validation_dataset._convert_to_torch()


writer = SummaryWriter()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # First fully connected layer
        self.fc1 = nn.Linear(3, 50)
        # Second fully connected layer that outputs our 1 output channel (image)
        self.fc2 = nn.Linear(50, 30)
        self.fc3 = nn.Linear(30, 10)
        self.fc4 = nn.Linear(10, 2)
        # self.fc3 = nn.Linear(5, 2)
        # self.sigmoid = nn.Sigmoid()
    # x represents our data

    def forward(self, x):
        compo = x
        compo = self.fc1(compo)
        compo = self.fc2(compo)
        compo = self.fc3(compo)
        compo = F.tanh(compo)
        # compo = F.relu(compo)
        compo = self.fc4(compo)
        # compo = F.relu(compo)
        # compo = self.fc3(compo)
        # output = F.relu(compo)
        # output = self.sigmoid(compo)

        return compo

    def predict(self, x):
        pred = F.softmax(self.forward(x))
        ans = []
        # Pick the class with maximum weight
        for t in pred:
            if t[0] > t[1]:
                ans.append(0)
            else:
                ans.append(1)
        return torch.tensor(ans)


net = Net()
# nn.BCELoss() #nn.BCEWithLogitsLoss() #nn.L1Loss()#
criterion = nn.CrossEntropyLoss()
# optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=5e-4)
#scheduler=ReduceLROnPlateau(optimizer,mode='min',patience=5, verbose=True)


no_epochs = 5000
for epoch in range(no_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    validation_loss = 0.0
    avg_t_loss = 0.0
    avg_v_loss = 0.0
    # get the inputs; data is a list of [inputs, labels]

    train_output = []
    # zero the parameter gradients

    for idx in range(len(input_images)):
        optimizer.zero_grad()
    # forward + backward + optimize
        outputs = net.forward(input_images[idx])
        train_output.append(outputs)
        t_loss = criterion(outputs, output_images[idx])  # tensor_output)
        # writer.add_scalar("Loss/train", t_loss, epoch)
        t_loss.backward()
        optimizer.step()
        running_loss += t_loss.item()

        if epoch % 200 == 0:
            print('%d EPOCH: [%d] loss: %.3f' %
                  (epoch, idx,  t_loss.item()))
    avg_t_loss = running_loss / len(input_images)

    for idx in range(len(validation_input)):
        # forward + backward + optimize
        outputs = net.forward(validation_input[idx])
        v_loss = criterion(outputs, validation_output[idx])

        validation_loss += v_loss.item()
    avg_v_loss = validation_loss / len(validation_input)

    writer.add_scalars(
        "Loss/epoch", {'valid': avg_v_loss, 'train': avg_t_loss}, epoch)

writer.flush()
# outputs = net.predict(torch.unsqueeze(input_images[0],1))
# print(outputs, output_images[0])
print('Finished Training')
PATH = './threshold.pth'
torch.save(net.state_dict(), PATH)
