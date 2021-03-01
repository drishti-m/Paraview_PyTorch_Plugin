import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

input_image = np.arange(-20.50, 20.51234, 0.01)
in_2 = np.arange(-20.51, 20.51234, 0.04)
in_3 = np.arange(-80.5134, 80.51234, 0.1)
in_4 = np.arange(-100.134, 100.51234, 0.09)
in_5 = np.arange(0.5, 300.5, 0.1)

print(input_image.shape)

output_image = input_image > 0.0
output_image = output_image.astype(float)
out_2 = in_2 > 0.0
out_2 = out_2.astype(float)
out_3 = in_3 > 0.0
out_3 = out_3.astype(float)
out_4 = in_4 > 0.0
out_4 = out_4.astype(float)
out_5 = in_5 > 0.0
out_5 = out_5.astype(float)
print(output_image.shape)

tensor_input = torch.from_numpy(input_image).to(torch.float)
tensor_output = torch.from_numpy(output_image).to(torch.float)
ti_i2 = torch.from_numpy(in_2).to(torch.float)
ti_i3 = torch.from_numpy(in_3).to(torch.float)
ti_i4 = torch.from_numpy(in_4).to(torch.float)
ti_i5 = torch.from_numpy(in_5).to(torch.float)
ti_o2 = torch.from_numpy(out_2).to(torch.float)
ti_o3 = torch.from_numpy(out_3).to(torch.float)
ti_o4 = torch.from_numpy(out_4).to(torch.float)
ti_o5 = torch.from_numpy(out_5).to(torch.float)
# print(ti_o4)
# print(ti_o5)
# print(ti_o2)
# print(ti_o3)
# print(tensor_output)
# create neural network


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # First fully connected layer
        self.fc1 = nn.Linear(1, 2000)
        # Second fully connected layer that outputs our 1 output channel (image)
        self.fc2 = nn.Linear(2000, 100)
        self.fc3 = nn.Linear(100, 1)

    # x represents our data
    def forward(self, x):
        compo = x
        compo = self.fc1(compo)
        compo = F.relu(compo)
        compo = self.fc2(compo)
        compo = F.relu(compo)
        compo = self.fc3(compo)
        output = F.relu(compo)

        return output


net = Net()
tensor_input = torch.unsqueeze(tensor_input, 1)
tensor_output = torch.unsqueeze(tensor_output, 1)
ti_i2 = torch.unsqueeze(ti_i2, 1)
ti_i3 = torch.unsqueeze(ti_i3, 1)
ti_i4 = torch.unsqueeze(ti_i4, 1)
ti_i5 = torch.unsqueeze(ti_i5, 1)
ti_o2 = torch.unsqueeze(ti_o2, 1)
ti_o3 = torch.unsqueeze(ti_o3, 1)
ti_o4 = torch.unsqueeze(ti_o4, 1)
ti_o5 = torch.unsqueeze(ti_o5, 1)


net.forward(tensor_input)

criterion = nn.L1Loss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

for epoch in range(50):  # loop over the dataset multiple times

    running_loss = 0.0
    # get the inputs; data is a list of [inputs, labels]

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(tensor_input)
    loss = criterion(outputs, tensor_output)  # tensor_output)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
    print('[%d] loss: %.3f' %
          (1,  loss.item()))
    optimizer.zero_grad()

    outs_2 = net(ti_i2)
    loss = criterion(outs_2, ti_o2)  # tensor_output)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
    print('[%d] loss: %.3f' %
          (2,  loss.item()))
    optimizer.zero_grad()

    outs_3 = net(ti_i3)
    loss = criterion(outs_3, ti_o3)  # tensor_output)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
    print('[%d] loss: %.3f' %
          (3,  loss.item()))
    optimizer.zero_grad()

    outs_4 = net(ti_i4)
    loss = criterion(outs_4, ti_o4)  # tensor_output)
    loss.backward()
    optimizer.step()
    print('[%d] loss: %.3f' %
          (4,  loss.item()))
    optimizer.zero_grad()

    outs_5 = net(ti_i5)
    loss = criterion(outs_5, ti_o5)  # tensor_output)
    loss.backward()
    optimizer.step()
    print('[%d] loss: %.3f' %
          (5,  loss.item()))
    optimizer.zero_grad()

    # print statistics
    running_loss += loss.item()
    # if i % 2 == 0:    # print every 2000 mini-batches
    # print('[%d] loss: %.3f' %
    #       (epoch + 1,  running_loss))
    # running_loss = 0.0

print('Finished Training')
print(outs_5.detach().numpy(), outs_4.detach().numpy(), outs_3.detach(
).numpy(), outs_2.detach().numpy(), outputs.detach().numpy())
PATH = './threshold.pth'
torch.save(net.state_dict(), PATH)
