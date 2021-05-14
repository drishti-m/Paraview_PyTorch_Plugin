"""
This is code for training data for binary segmentation based on whether
each cell's magnitude value is greater than threshold value or not.
Here, threshold value = 0.01
Training inspired by tutorial:
https://medium.com/@prudhvirajnitjsr/simple-classifier-using-pytorch-37fba175c25c
"""
from os.path import dirname
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os


class Dataset:
    """
    Class for Dataset with input data and expected output
    """

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
    """
    Read arrays from file at path, and separate arrays based on delimiter
    """
    arrays = np.genfromtxt(path, delimiter=delimiter)
    arrays = arrays[1:]
    arrays = arrays[:, 4:7]
    return arrays


def CreateDataset(all_arrays):
    """
    Create training dataset from all_arrays
    """
    dataset = Dataset()

    dataset._addData(all_arrays[0])
    dataset._addData(all_arrays[1])
    dataset._addData(all_arrays[2])
    dataset._addData(all_arrays[4])
    dataset._addData(all_arrays[7])
    dataset._addData(all_arrays[8])
    dataset._addData(all_arrays[9])
    dataset._addData(all_arrays[11])
    dataset._addData(all_arrays[12])

    return dataset


def CreateValidationDataset(all_arrays):
    """
    Create validation dataset from all_arrays
    """
    validation_dataset = Dataset()
    validation_dataset._addData(all_arrays[3])
    validation_dataset._addData(all_arrays[5])
    validation_dataset._addData(all_arrays[6])
    validation_dataset._addData(all_arrays[10])
    return validation_dataset


class Net(nn.Module):
    """
    Architecture for neural net model
    """

    def __init__(self):
        super(Net, self).__init__()

        # First fully connected layer
        self.fc1 = nn.Linear(3, 80)
        # Second fully connected layer
        self.fc2 = nn.Linear(80, 40)
        # Third fully connected layer
        self.fc3 = nn.Linear(40, 10)
        # Fourth fully connected layer
        self.fc4 = nn.Linear(10, 2)

    # x represents our data
    def forward(self, x):
        compo = x
        compo = self.fc1(compo)
        compo = self.fc2(compo)
        compo = self.fc3(compo)
        compo = torch.tanh(compo)
        compo = self.fc4(compo)

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
        return torch.tensor(ans).to(torch.float)


def main():
    # load datasets
    all_arrays = []
    dir_name = os.path.dirname(os.path.realpath(__file__))
    for i in range(0, 13, 1):
        path = dir_name + '/datasets/velocity/velocity_data_' + \
            str(i) + ".csv"
        print(path)
        current_array = read_arrays(path)
        all_arrays.append(current_array)
    all_arrays = np.array(all_arrays)

    # split training and validation dataset
    dataset = CreateDataset(all_arrays)
    input_images = dataset.input_data
    output_images = dataset._output_array()
    input_images = dataset._convert_to_torch()
    validation_dataset = CreateValidationDataset(all_arrays)
    validation_input = validation_dataset.input_data
    validation_output = validation_dataset._output_array()
    validation_input = validation_dataset._convert_to_torch()

    writer = SummaryWriter()

    # define params for neural net model
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=5e-4)
    no_epochs = 500

    # loop over the dataset multiple times
    for epoch in range(no_epochs):

        running_loss = 0.0
        validation_loss = 0.0
        avg_t_loss = 0.0
        avg_v_loss = 0.0

        # training dataset
        train_output = []
        for idx in range(len(input_images)):
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net.forward(input_images[idx])
            train_output.append(outputs)
            t_loss = criterion(outputs, output_images[idx])
            t_loss.backward()
            optimizer.step()
            running_loss += t_loss.item()

            if epoch % 200 == 0:
                print('%d EPOCH: [%d] training loss: %.3f' %
                      (epoch, idx,  t_loss.item()))
        avg_t_loss = running_loss / len(input_images)

        # validation dataset
        for idx in range(len(validation_input)):
            outputs = net.forward(validation_input[idx])
            v_loss = criterion(outputs, validation_output[idx])
            validation_loss += v_loss.item()
            if epoch % 200 == 0:
                print('%d EPOCH: [%d] validation loss: %.3f' %
                      (epoch, idx,  v_loss.item()))
        avg_v_loss = validation_loss / len(validation_input)

        writer.add_scalars(
            "Loss/epoch", {'valid': avg_v_loss, 'train': avg_t_loss}, epoch)

    writer.flush()

    # save training model
    print('Finished Training')
    PATH = dir_name + '/models/fluid-segment(velocity).pth'
    torch.save(net.state_dict(), PATH)
    print("Model saved at ", PATH)


if __name__ == "__main__":
    main()
