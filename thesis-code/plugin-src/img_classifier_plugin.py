import torch
import torchvision
import torchvision.transforms as transforms
import os
from paraview.vtk.util import numpy_support as ns
from paraview import simple
from paraview.util.vtkAlgorithm import *
import vtk
import numpy as np


@smproxy.filter()
@smproperty.input(name="classifier", port_index=0)
@smdomain.datatype(dataTypes=["vtkImageData", "vtkRectilinearGrid"], composite_data_supported=True)
class ML_Classifier(VTKPythonAlgorithmBase):
    input_data_type = ""
    t_port = 0
    t_index = 0
    model_path = ""
    label_path = ""

    def __init__(self):
        VTKPythonAlgorithmBase.__init__(
            self, nInputPorts=1, nOutputPorts=1, outputType="vtkTable")

    def FillInputPortInformation(self, port, info):
        # self = vtkPythonAlgorithm
        if port == 0:
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
            input_vtk_array, x, y, z = self.Get_Input_Array(inInfoVec, "image")

        elif self.input_data_type == "vtkRectilinearGrid":
            input_vtk_array, x, y, z = self.Get_Input_Array(inInfoVec, "recti")

        predictions_vtk, strArray = self.Classify_Image(
            input_vtk_array, x, y, z)

        out = vtkTable.GetData(outInfoVec, 0)
        dsa = out.GetRowData()
        dsa.AddArray(predictions_vtk)
        dsa.AddArray(strArray)
        return 1

    def Get_Input_Array(self, inInfoVec, i_type):
        from vtkmodules.vtkCommonDataModel import vtkRectilinearGrid, vtkImageData
        from vtkmodules.vtkCommonCore import VTK_DOUBLE
        if i_type == "image":
            pdi = vtkImageData.GetData(inInfoVec[0], 0)
        elif i_type == "recti":
            pdi = vtkRectilinearGrid.GetData(inInfoVec[0], 0)
        x, y, z = pdi.GetDimensions()
        print("Shape of input vtk Image: ", x, y, z)
        pixels_vtk_array = pdi.GetPointData().GetAbstractArray(0)
        return pixels_vtk_array, x, y, z

    def Pre_Process_Image(self, img):
        transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize(256),
                                        # transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])

        img_t = transform(img)
        return img_t

    def Get_Model_Labels(self, labels_path):
        with open(labels_path) as f:
            classes = [line.strip() for line in f.readlines()]
        return classes

    def Make_Predictions(self, out, classes):
        _, index = torch.max(out, 1)

        percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

        #print(classes[index[0]], percentage[index[0]].item())
        _, indices = torch.sort(out, descending=True)
        predictions_np = [percentage[idx].item()
                          for idx in indices[0][:10]]
        predicted_classes = [classes[idx]
                             for idx in indices[0][:10]]
        return predictions_np, predicted_classes
        # print([(classes[idx], percentage[idx].item())
        #        for idx in indices[0][:5]])

    def Create_vtkTable_Columns(self, predictions_np, predicted_classes):
        predictions_vtk = ns.numpy_to_vtk(predictions_np)
        predictions_vtk.SetName("Predicted Values")
        strArray = vtk.vtkStringArray()

        strArray.SetName("Label names")
        strArray.SetNumberOfTuples(len(predicted_classes))
        for i in range(0, len(predicted_classes), 1):
            strArray.SetValue(i, predicted_classes[i])
        return predictions_vtk, strArray

    def Forward_Pass(self, img_t, net):
        batch_t = torch.unsqueeze(img_t, 0)
        out = net(batch_t)
        return batch_t, out

    def Classify_Image(self, pixels_vtk_array, x, y, z):
        self.labels_path = "./pytorch_data/imagenet_classes.txt"
        self.model_path = './pytorch_data/alexnet.pth'

        pixels_np_array = self.convert_vtk_to_numpy(pixels_vtk_array, x, y)
        print("Converted vtk array to suitable numpy representation with shape: ",
              pixels_np_array.shape)

        print("Loading model at path: ", self.model_path)
        net = torch.load(self.model_path)
        net.eval()

        img_t = self.Pre_Process_Image(pixels_np_array)
        batch_t, out = self.Forward_Pass(img_t, net)
        classes = self.Get_Model_Labels(self.labels_path)
        predictions_np, predicted_classes = self.Make_Predictions(out, classes)
        predictions_vtk, strArray = self.Create_vtkTable_Columns(
            predictions_np, predicted_classes)

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
        if pixels_vtk_array.GetNumberOfComponents() < 3:
            pixels_np_array = np.repeat(pixels_np_array, 3)
        # x, y reversed between vtk <-> numpy array
        pixels_np_array = pixels_np_array.reshape((y, x, 3))
        #pixels_np_array = np.flip(pixels_np_array, axis=2)
        pixels_np_array = np.flip(pixels_np_array, axis=0)
        pixels_np_array = pixels_np_array.astype(np.uint8)

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

    @smproperty.stringvector(name="Class Labels Path")
    def SetLabelsPathR(self, x):
        print("Labels path: ", x)
        self.labels_path = x
        self.Modified()
