#from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from paraview.util.vtkAlgorithm import *
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import os
import vtk
from vtk.util import numpy_support
from paraview.vtk.util import numpy_support as ns
import matplotlib.pyplot as plt
from paraview import simple
import numpy as np


@smproxy.filter()
@smproperty.input(name="InputTable", port_index=0)
@smdomain.datatype(dataTypes=["vtkImageData"], composite_data_supported=False)
# @smproperty.input(name="InputDataset", port_index=0)
# @smdomain.datatype(dataTypes=["vtkDataSet"], composite_data_supported=False)
class CifarClassifier(VTKPythonAlgorithmBase):
    def __init__(self):
        VTKPythonAlgorithmBase.__init__(
            self, nInputPorts=1, nOutputPorts=1, outputType="vtkTable")
        x = 0
        y = 0
        z = 0

    def FillInputPortInformation(self, port, info):
        print("port info set")
        if port == 0:
            info.Set(self.INPUT_REQUIRED_DATA_TYPE(), "vtkImageData")
            print(info)
        # else:
        #     info.Set(self.INPUT_REQUIRED_DATA_TYPE(), "vtkTable")
        return 1

    def RequestData(self, request, inInfoVec, outInfoVec):
        from vtkmodules.vtkCommonDataModel import vtkImageData, vtkTable, vtkDataSet, vtkPolyData
        from vtkmodules.vtkCommonCore import VTK_DOUBLE
        print(request, "req")
        print("self:", self.x, self.y, self.z)
        # PLUGIN different PART: pdi = vtkImagedata.GetData(inInfoVec[portNumber],0) instead of
        # prog filter: pdi = self.GetInput()
        pdi = vtkImageData.GetData(inInfoVec[0], 0)
        no_arrays = pdi.GetPointData().GetNumberOfArrays()
        float_array = pdi.GetPointData().GetAbstractArray(0)
        numpy_array = ns.vtk_to_numpy(float_array)
        numpy_array = numpy_array.reshape((3, 32, 32), order="F")
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
        # print(torch_array)
        torch_list = torch.cat(
            (torch_array, torch_array, torch_array, torch_array))
        torch_list = torch.reshape(torch_list, (4, 3, 32, 32))
        # print(torch_list)
        outputs = net(torch_list)
        # print(outputs)
        _, predicted = torch.max(outputs, 1)
        classes = np.array(['plane', 'car', 'bird', 'cat',
                            'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])

        print('\n Model Predicted the image as: ', classes[predicted[0]])
        # print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
        #                              for j in range(4)))

        predictions_np = outputs[0].detach().numpy()
        sorted_idx = np.flip(np.argsort(predictions_np))
        predictions_np = np.flip(np.sort(predictions_np))

        print('sorted idx; ', sorted_idx)
        predictions_vtk = ns.numpy_to_vtk(predictions_np)
        predictions_vtk.SetName("Predicted Values")
        print(predictions_vtk)
        print(float_array)

        # PLUGIN different part: output = GetData() instead of self.GetOutput()
        output = vtkTable.GetData(outInfoVec, 0)
        dsa = output.GetRowData()
        dsa.AddArray(predictions_vtk)
        strArray = vtk.vtkStringArray()

        strArray.SetName("Label names")
        sorted_classes = classes[sorted_idx]
        strArray.SetNumberOfTuples(len(sorted_classes))
        for i in range(len(sorted_classes)-1, -1, -1):
            strArray.SetValue(i, sorted_classes[i])
        dsa.AddArray(strArray)

        # input1 = vtkDataSet.GetData(inInfoVec[1], 0)
        # output = vtkTable.GetData(outInfoVec, 0)
        # do work

        print("Pretend work done!")
        return 1

    @smproperty.xml("""
        <DoubleVectorProperty name="Center"
            number_of_elements="3"
            default_values="0 0 0"
            command="SetCenter">
            <DoubleRangeDomain name="range" />
            <Documentation>Set center of the superquadric</Documentation>
        </DoubleVectorProperty>""")
    def SetCenter(self, x, y, z):
        #self._realAlgorithm.SetCenter(x, y, z)
        # self.Modified()
        # print(self)
        print("center set", x, y, z)
        self.x = x
        self.y = y
        self.z = z


@smproxy.filter()
@smproperty.input(name="InputTable", port_index=0)
@smdomain.datatype(dataTypes=["vtkTable"], composite_data_supported=False)
# @smproperty.input(name="InputDataset", port_index=0)
# @smdomain.datatype(dataTypes=["vtkDataSet"], composite_data_supported=False)
class ExampleTwoInputFilter(VTKPythonAlgorithmBase):
    def __init__(self):
        VTKPythonAlgorithmBase.__init__(
            self, nInputPorts=1, nOutputPorts=1, outputType="vtkPolyData")

    def FillInputPortInformation(self, port, info):
        if port == 0:
            #     info.Set(self.INPUT_REQUIRED_DATA_TYPE(), "vtkDataSet")
            # else:
            info.Set(self.INPUT_REQUIRED_DATA_TYPE(), "vtkTable")
        return 1

    def RequestData(self, request, inInfoVec, outInfoVec):
        from vtkmodules.vtkCommonDataModel import vtkTable, vtkDataSet, vtkPolyData
        input0 = vtkTable.GetData(inInfoVec[0], 0)
        # input1 = vtkDataSet.GetData(inInfoVec[1], 0)
        output = vtkPolyData.GetData(outInfoVec, 0)
        # do work
        print("filter example Pretend work done!")
        return 1
