import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import os
from vtk.util import numpy_support
from paraview.vtk.util import numpy_support as ns
import matplotlib.pyplot as plt
from paraview import simple
import numpy as np


pdi = self.GetInput()
#print(pdi)
#print(pdi.GetCellPoint(x, y))
no_arrays = pdi.GetPointData().GetNumberOfArrays()
float_array = pdi.GetPointData().GetAbstractArray(0)
#print(pdi.GetPointData())
#help(self)
#print(float_array)
numpy_array = ns.vtk_to_numpy(float_array)
#print(numpy_array.shape)
numpy_array = numpy_array.reshape((3,32,32), order = "F")
#print("NUMPY ARAY")
#print(numpy_array)
print(os.getcwd())
PATH = './pytorch_data/cifar_net.pth'



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
print("Loading pre-trained model 'cifar_net.pth'..")
net = Net()
net.load_state_dict(torch.load(PATH))


torch_array = torch.from_numpy(numpy_array)
#print(torch_array)
torch_list = torch.cat((torch_array, torch_array, torch_array, torch_array))
torch_list = torch.reshape(torch_list, (4,3,32,32))
#print(torch_list)
outputs = net(torch_list)
#print(outputs)
_, predicted = torch.max(outputs, 1)
classes = np.array(['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])

print('\n Model Predicted the image as: ',classes[predicted[0]])
#print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
#                              for j in range(4)))

predictions_np = outputs[0].detach().numpy()
sorted_idx = np.flip(np.argsort(predictions_np))
predictions_np = np.flip(np.sort(predictions_np))

print('sorted idx; ', sorted_idx)
predictions_vtk = ns.numpy_to_vtk(predictions_np)
predictions_vtk.SetName("Predicted Values")
print(predictions_vtk)
print(float_array)
out = self.GetOutput()
dsa = out.GetRowData();
dsa.AddArray(predictions_vtk)
strArray = vtk.vtkStringArray()

strArray.SetName("Label names")
sorted_classes = classes[sorted_idx]
strArray.SetNumberOfTuples(len(sorted_classes))
for i in range(len(sorted_classes)-1, -1, -1):
    strArray.SetValue(i, sorted_classes[i])
dsa.AddArray(strArray)




