# working binary classifier
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

    def _addData(self, elements):
        #t_i = torch.from_numpy(elements).to(torch.float)
        self.input_data.append(elements)
        return self.input_data

    def _output_array(self):
        for i_array in self.input_data:
            output = i_array > 0.0000
            output = output.astype(float)
            output = torch.from_numpy(output).to(torch.long)
            self.output_data.append(output)
        return self.output_data

    def _convert_to_torch(self):
        for idx in range(len(self.input_data)):
            self.input_data[idx] = torch.from_numpy(
                self.input_data[idx]).to(torch.float)

        return self.input_data


dataset = Dataset()
dataset._addData(np.arange(-2.0102, 2.0104,  0.051))
dataset._addData(np.arange(-2.5103, 2.5123, 0.0456))
dataset._addData(np.arange(-8.5134, 8.51234, 0.1109))
dataset._addData(np.arange(-1.1344, 10.51234, 0.0910))
dataset._addData(np.arange(0.5000, 12.51234, 0.1234))
dataset._addData(np.array([-0.0212228]))
dataset._addData(np.arange(-20.50, 20.51234, 0.011113))
dataset._addData(np.arange(-20.51, 20.51234, 0.04))
dataset._addData(np.arange(-80.5134, 80.51234, 0.1))
dataset._addData(np.arange(-10.134, 10.51234, 0.09))
dataset._addData(np.arange(0.5, 30.5, 0.1))
dataset._addData(np.arange(-0.5, 30.5, 0.1))
dataset._addData(np.arange(1.556, 45.67899, 0.1))
dataset._addData(np.arange(10.0, 145.67899, 0.134))
dataset._addData(np.arange(100.1345, 300.9876, 1.567))
dataset._addData(np.arange(0.000, 23.9560, 0.0123))
dataset._addData(np.arange(-3.0345, -1.0345, 0.0134))
input_images = dataset.input_data
output_images = dataset._output_array()
input_images = dataset._convert_to_torch()


writer = SummaryWriter()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # First fully connected layer
        self.fc1 = nn.Linear(1, 5)
        # Second fully connected layer that outputs our 1 output channel (image)
        self.fc2 = nn.Linear(5, 2)
        # self.fc3 = nn.Linear(5, 2)
        # self.sigmoid = nn.Sigmoid()
    # x represents our data

    def forward(self, x):
        compo = x
        compo = self.fc1(compo)
        compo = F.tanh(compo)
        # compo = F.relu(compo)
        compo = self.fc2(compo)
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


no_epochs = 2000
for epoch in range(no_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    # get the inputs; data is a list of [inputs, labels]

    train_output = []
    # zero the parameter gradients

    for idx in range(len(input_images)):
        optimizer.zero_grad()
    # forward + backward + optimize
        outputs = net.forward(torch.unsqueeze(input_images[idx], 1))
        #print(outputs )
        train_output.append(outputs)
        loss = criterion(outputs, output_images[idx])  # tensor_output)
        writer.add_scalar("Loss/train", loss, epoch)
        loss.backward()
        optimizer.step()
        # scheduler.step(loss.item())
        running_loss += loss.item()
        print('[%d] loss: %.3f' %
              (idx,  loss.item()))
writer.flush()

# outputs = net.predict(torch.unsqueeze(input_images[0],1))
# print(outputs, output_images[0])
print('Finished Training')
PATH = './pressure_threshold.pth'
torch.save(net.state_dict(), PATH)
