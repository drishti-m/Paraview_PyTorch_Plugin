# working binary classifier for max pressure data
# ref: https://medium.com/@prudhvirajnitjsr/simple-classifier-using-pytorch-37fba175c25c
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Dataset:
    def __init__(self):
        self.input_data = []
        self.output_data = []
        self.total_data_amt = 0.0

    def _addData(self, elements):
        #t_i = torch.from_numpy(elements).to(torch.float)
        self.input_data.append(elements)
        self.total_data_amt += len(elements)
        return self.input_data

    def _output_array(self):
        for i_array in self.input_data:
            output = np.max(i_array)
            output = output > 5.0
            output = output.astype(float)
            output = np.array([output, output])
            output = torch.from_numpy(output).to(torch.long)
            self.output_data.append(output)
        return self.output_data

    def _convert_to_torch(self):
        for idx in range(len(self.input_data)):
            self.input_data[idx] = torch.from_numpy(
                self.input_data[idx]).to(torch.float)

        return self.input_data


def read_arrays(path, delimiter=','):
    arrays_19 = np.genfromtxt(path, delimiter=delimiter)
    arrays_19 = arrays_19[1:]
    arrays_19 = arrays_19[:, 4]

    return arrays_19


all_arrays = []
for i in range(0, 10, 1):
    path = 'pressure_data_' + str(i) + ".csv"
    print(path)
    current_array = read_arrays(path)
    # print (np.array(current_array).flatten())
    all_arrays.append(np.array(current_array).flatten())
all_arrays = np.array(all_arrays)

dataset = Dataset()

dataset._addData(all_arrays[0])
dataset._addData(all_arrays[1])
dataset._addData(all_arrays[3])
dataset._addData(all_arrays[5])
dataset._addData(all_arrays[6])
dataset._addData(all_arrays[9])
dataset._addData(all_arrays[8])

input_images = dataset.input_data
output_images = dataset._output_array()
input_images = dataset._convert_to_torch()

validation_dataset = Dataset()
validation_dataset._addData(all_arrays[2])
validation_dataset._addData(all_arrays[4])
validation_dataset._addData(all_arrays[7])

validation_input = validation_dataset.input_data
validation_output = validation_dataset._output_array()
validation_input = validation_dataset._convert_to_torch()
print(all_arrays[0].shape)

writer = SummaryWriter()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        batch_size = 2500
        # First fully connected layer
        self.fc1 = nn.Linear(batch_size, 50)
        # Second fully connected layer that outputs our 1 output channel (image)
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, 2)
        # self.fc3 = nn.Linear(5, 2)
        # self.sigmoid = nn.Sigmoid()
    # x represents our data

    def forward(self, x):
        compo = x
        compo = self.fc1(compo)
        compo = self.fc2(compo)
        compo = F.tanh(compo)
        # compo = F.relu(compo)
        compo = self.fc3(compo)
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


no_epochs = 500
for epoch in range(no_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    validation_loss = 0.0
    avg_t_loss = 0.0
    avg_v_loss = 0.0
    # get the inputs; data is a list of [inputs, labels]

    train_output = []
    # zero the parameter gradients

    for idx in range(0, len(input_images)):
        optimizer.zero_grad()
    # forward + backward + optimize
        # torch.unsqueeze(input_images[idx],1)
        t_input = torch.stack([input_images[idx], input_images[idx]], dim=0)
        print(t_input.shape)
        outputs = net.forward(t_input)
        train_output.append(outputs)
        t_loss = criterion(outputs, output_images[idx])  # tensor_output)
        # writer.add_scalar("Loss/train", t_loss, epoch)
        t_loss.backward()
        optimizer.step()
        # scheduler.step(loss.item())
        running_loss += t_loss.item()

        print('[%d] loss: %.3f' %
              (idx,  t_loss.item()))
    avg_t_loss = running_loss / len(input_images)

    for idx in range(len(validation_input)):
        # forward + backward + optimize
        # torch.unsqueeze(input_images[idx],1)
        v_input = torch.stack(
            [validation_input[idx], validation_input[idx]], dim=0)
        print(v_input.shape)
        outputs = net.forward(v_input)
        v_loss = criterion(outputs, validation_output[idx])

        # scheduler.step(loss.item())
        validation_loss += v_loss.item()
    avg_v_loss = validation_loss / len(validation_input)

    writer.add_scalars(
        "Loss/epoch", {'valid': avg_v_loss, 'train': avg_t_loss}, epoch)

writer.flush()
a = np.arange(0, 2500, 1)
a = torch.from_numpy(a).to(torch.float)
outputs = net.predict(torch.stack([a, a], dim=0))
print(outputs)
print('Finished Training')
PATH = './threshold.pth'
torch.save(net.state_dict(), PATH)
