"""
This is the plugin code for fluid classifier (for scalar data eg, pressure).
Accepted input data models: VTK Rectilinear Grid
Output data model: VTK Table
It takes from user two parameters: Trained Model's Path, Model's Class Defn Path.
The path can be either absolute or relative to Paraview's binary executable location.
This plugin is designed to classify a Grid as "High" or "Low" depending on whether 
the max value of a grid is greater or lesser than the threshold value respectively.
Threshold value is determined during training of model.

"""

from paraview.util.vtkAlgorithm import *
from paraview.vtk.util import numpy_support as ns
from paraview import simple
import vtk
from vtkmodules.numpy_interface import dataset_adapter as DA
import torch
import torch.nn as nn
import os
import numpy as np


@smproxy.filter()
@smproperty.input(name="InputRectilinear", port_index=0)
@smdomain.datatype(dataTypes=["vtkRectilinearGrid"], composite_data_supported=True)
class ML_Fluid_Classifier(VTKPythonAlgorithmBase):
    input_data_type = ""
    t_port = 0
    t_index = 0
    model_path = ""
    class_path = ""

    def __init__(self):
        VTKPythonAlgorithmBase.__init__(
            self, nInputPorts=1, nOutputPorts=1, outputType="vtkTable")

    # First step in pipeline: set Port info
    def FillInputPortInformation(self, port, info):
        if port == 0:
            self.t_port = port
            self.t_index = info.Get(self.INPUT_CONNECTION())
        print("Port info set")
        return 1

    # Request Data from input and create output
    def RequestData(self, request, inInfoVec, outInfoVec):
        from vtkmodules.vtkCommonDataModel import vtkRectilinearGrid, vtkImageData, vtkTable
        from vtkmodules.vtkCommonCore import VTK_DOUBLE

        self.input_data_type = self.GetInputDataObject(
            self.t_port, self.t_index).GetClassName()
        if self.input_data_type == "vtkRectilinearGrid":
            predictions_vtk, strArray, x, y, z, xCoords, yCoords, zCoords = self.Process_RectlinearGrid(
                inInfoVec, outInfoVec)

        out = vtkTable.GetData(outInfoVec, 0)
        dsa = out.GetRowData()
        dsa.AddArray(predictions_vtk)
        dsa.AddArray(strArray)
        return 1

    def Create_vtkTable_Columns(self, predictions_np, predicted_classes):
        """
        Helper function to create rows and columns for table

        Args:
        predictions_np: numpy array of predicted confidence values
        predicted_classes: string array of corresponding classes

        Returns:
        predictions_vtk: VTK array for predicted confidence values 
        strArray: VTK array of corresponding classes
        """
        predictions_vtk = ns.numpy_to_vtk(predictions_np)
        predictions_vtk.SetName("Confidence %")
        strArray = vtk.vtkStringArray()

        strArray.SetName("Predicted Labels")
        strArray.SetNumberOfTuples(len(predicted_classes))
        for i in range(0, len(predicted_classes), 1):
            strArray.SetValue(i, predicted_classes[i])
        return predictions_vtk, strArray

    def Process_RectlinearGrid(self, inInfoVec, outInfoVec):
        """
        Helper function to process info from input to send it for model inference

        Args:
        inInfoVec: input information vector
        outInfoVec: output information vector

        Returns:
        predictions_vtk: VTK array for predicted confidence values 
        strArray: VTK array of corresponding classes
        x, y, z: output x,y,z dimensions respectively
        xCoords, yCoords, zCoords: output's x co-ordinates, y-coordinates, z-coordinates respectively
        output_vtk_array: VTK array for output
        """
        from vtkmodules.vtkCommonDataModel import vtkRectilinearGrid, vtkImageData
        from vtkmodules.vtkCommonCore import VTK_DOUBLE
        pdi = vtkRectilinearGrid.GetData(inInfoVec[0], 0)

        x, y, z = pdi.GetDimensions()
        xCoords = pdi.GetXCoordinates()
        yCoords = pdi.GetYCoordinates()
        zCoords = pdi.GetZCoordinates()
        no_arrays = pdi.GetPointData().GetNumberOfArrays()
        input_vtk_array = pdi.GetPointData().GetAbstractArray(0)
        no_components = input_vtk_array.GetNumberOfComponents()
        input_numpy_array = ns.vtk_to_numpy(input_vtk_array)

        predictions_np, predicted_classes = self.Classify(
            input_numpy_array)
        predictions_vtk, strArray = self.Create_vtkTable_Columns(
            predictions_np, predicted_classes)

        return predictions_vtk, strArray, x, y, z, xCoords, yCoords, zCoords

    def Classify(self, numpy_array):
        """
        Helper function to pre-process input and feed it to pre-trained model.

        Args:
        numpy_array: Numpy array of input to model


        Returns:
        Params returned by Function: Make_Predictions
        """
        from importlib import import_module

        if self.class_path:
            module_name = self.get_trained_class_module_name()
            module = import_module(module_name)
            net = module.Net()
            net.load_state_dict(torch.load(self.model_path))
        else:
            net = torch.load(self.model_path)

        print(numpy_array.shape)
        torch_array = torch.from_numpy(numpy_array.copy()).to(torch.float)
        torch_array = torch.stack([torch_array, torch_array], dim=0)
        print(torch_array.shape)

        return self.Make_Predictions(net, torch_array)

    def Make_Predictions(self, net, torch_array):
        """
        Helper function to feed input to pre-trained model.

        Args:
        net: Neural network model
        torch_array: input for model as Tensor array


        Returns:
        predictions_np: Predicted confidence values as Numpy array
        predicted_classes: Corresponding class labels
        """
        classes = ["LOW", "HIGH"]
        o_tensors = net(torch_array)
        percentages = torch.nn.functional.softmax(o_tensors, dim=1)[0] * 100
        _, indices = torch.sort(o_tensors[0], descending=True)
        predictions_np = [percentages[idx].item()
                          for idx in indices]

        predicted_classes = [classes[idx]
                             for idx in indices]

        print("Prediction: ",  predicted_classes[0], ": ", predictions_np[0])

        return predictions_np, predicted_classes

    def get_trained_class_module_name(self):
        """
        Get module name of class definition(architecture) of the trained model
        """
        import sys
        import os.path
        from pathlib import Path
        from importlib import import_module

        abs_path_class = os.path.abspath(self.class_path)
        class_dir = os.path.dirname(abs_path_class)
        sys.path.append(class_dir)
        module_name = Path(self.class_path).stem
        return module_name

    @ smproperty.stringvector(name="Trained Model's Path")
    def SetModelPathR(self, x):
        print("Model path: ", x)
        self.model_path = x
        self.Modified()

    @ smproperty.stringvector(name="Model's Class Defn Path")
    def SetClassPath(self, x):
        print("Class path: ", x)
        self.class_path = x
        self.Modified()
