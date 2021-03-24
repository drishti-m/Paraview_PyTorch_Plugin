import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import os
from vtk.util import numpy_support
from paraview.vtk.util import numpy_support as ns
import matplotlib.pyplot as plt
from paraview import simple
from paraview.util.vtkAlgorithm import *
import vtk
import numpy as np


@smproxy.filter()
@smproperty.input(name="classifier", port_index=0)
@smdomain.datatype(dataTypes=["vtkImageData"], composite_data_supported=True)
class ML_Classifier(VTKPythonAlgorithmBase):
    input_data_type = ""
    t_port = 0
    t_index = 0
    model_path = ""

    def __init__(self):
        VTKPythonAlgorithmBase.__init__(
            self, nInputPorts=1, nOutputPorts=1, outputType="vtkTable")

    def FillInputPortInformation(self, port, info):
        # self = vtkPythonAlgorithm
        if port == 0:

            # info.Set(self.INPUT_REQUIRED_DATA_TYPE(),
            #          "vtkRectilinearGrid")
            self.t_port = port
            self.t_index = info.Get(self.INPUT_CONNECTION())  # connection

        print("port info set")
        return 1

    def RequestData(self, request, inInfoVec, outInfoVec):
        from vtkmodules.vtkCommonDataModel import vtkTable, vtkImageData
        from vtkmodules.vtkCommonCore import VTK_DOUBLE

        self.input_data_type = self.GetInputDataObject(
            self.t_port, self.t_index).GetClassName()

        if self.input_data_type == "vtkImageData":
            predictions_vtk, strArray = self.Classify_Image(
                inInfoVec)

        out = vtkTable.GetData(outInfoVec, 0)
        dsa = out.GetRowData()
        dsa.AddArray(predictions_vtk)
        dsa.AddArray(strArray)
        return 1

    def Classify_Image(self, inInfoVec):
        from vtkmodules.vtkCommonDataModel import vtkRectilinearGrid, vtkImageData
        from vtkmodules.vtkCommonCore import VTK_DOUBLE
        labels_path = "./pytorch_data/imagenet_classes.txt"
        self.model_path = './pytorch_data/alexnet.pth'

        pdi = vtkImageData.GetData(inInfoVec[0], 0)
        x, y, z = pdi.GetDimensions()
        print("Shape of input vtk Image: ", x, y, z)
        pixels_vtk_array = pdi.GetPointData().GetAbstractArray(0)
        # print(pixels_vtk_array.GetTuple(216319))

        pixels_np_array = self.convert_vtk_to_numpy(pixels_vtk_array, x, y)
        print("Converted vtk array to suitable numpy representation with shape: ",
              pixels_np_array.shape)
        print(pixels_np_array)

        print("Loading model at path: ", self.model_path)
        net = torch.load(self.model_path)
        transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])

        img_t = transform(pixels_np_array)
        batch_t = torch.unsqueeze(img_t, 0)
        net.eval()
        out = net(batch_t)
        with open(labels_path) as f:
            classes = [line.strip() for line in f.readlines()]
        _, index = torch.max(out, 1)

        percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

        print(classes[index[0]], percentage[index[0]].item())
        _, indices = torch.sort(out, descending=True)
        predictions_np = [percentage[idx].item()
                          for idx in indices[0][:10]]
        predicted_classes = [classes[idx]
                             for idx in indices[0][:10]]
        print([(classes[idx], percentage[idx].item())
               for idx in indices[0][:5]])
        predictions_vtk = ns.numpy_to_vtk(predictions_np)
        predictions_vtk.SetName("Predicted Values")
        strArray = vtk.vtkStringArray()

        strArray.SetName("Label names")
        strArray.SetNumberOfTuples(len(predicted_classes))
        for i in range(0, len(predicted_classes), 1):
            strArray.SetValue(i, predicted_classes[i])
        # dsa.AddArray(strArray)
        return predictions_vtk, strArray

    def convert_vtk_to_numpy(self, pixels_vtk_array, x, y):
        """
        Adjusts proper order for vtk image array to convert into 
        equivalent numpy array.

        Args:
        pixels_vtk_array: vtk array 
        x: x-dimension of vtkImageData containing pixels_vtk_array
        y: y-dimension of vtkImageData containing pixels_vtk_array

        Returns:
        numpy array of shape (y,x,3) representing RGB values

        """
        pixels_np_array = ns.vtk_to_numpy(pixels_vtk_array)
        # print(pixels_np_array)

        # x, y reversed between vtk <-> numpy array
        pixels_np_array = pixels_np_array.reshape((y, x, 3))
        #pixels_np_array = np.flip(pixels_np_array, axis=2)
        pixels_np_array = np.flip(pixels_np_array, axis=0)

        return pixels_np_array

    def convert_numpy_to_vtk(self, rgb_array):
        """
        Adjusts proper order for numpy array to convert into 
        equivalent vtk array.

        Args:
        rgb_array: numpy array of shape (x,y,3)

        Returns:
        vtk array of shape (x*y, 3)

        """
        #r_x, r_y, r_z = rgb_array.shape
        r_x, r_y = rgb_array.shape
        r_z = 1
        rgb_array = np.flip(rgb_array)
        rgb_array = rgb_array.reshape((r_x*r_y, r_z))
        vtk_output = DA.numpyTovtkDataArray(rgb_array, name="segmented_pixels")
        # x, y reversed between vtk <-> numpy array
        return vtk_output, r_y, r_x, r_z

    @smproperty.stringvector(name="Trained Model Path")
    def SetModelPathR(self, x):
        print("Model path: ", x)
        self.model_path = x
        self.Modified()
