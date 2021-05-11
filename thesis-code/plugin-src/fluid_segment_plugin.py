  
"""
This is the plugin code for fluid segmentation.
Accepted input data models: VTK Rectilinear Grid
Output data model: VTK Rectilinear Grid
It takes from user two parameters: Trained Model's Path, Model's Class Defn Path.
The path can be either absolute or relative to Paraview's binary executable location.
This plugin is designed to segment a grid such that cells with magnitude greater than the threshold value are
labelled as 1, otherwise labelled as 0.
Threshold value is determined during training of model.
"""

from paraview.util.vtkAlgorithm import *
from paraview.vtk.util import numpy_support as ns
from paraview import simple
import vtk
from vtk.util import numpy_support
from vtkmodules.numpy_interface import dataset_adapter as DA
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np


@smproxy.filter()
@smproperty.input(name="InputRectilinear", port_index=0)
@smdomain.datatype(dataTypes=["vtkRectilinearGrid"], composite_data_supported=True)
class ML_Fluid_Segmentation(VTKPythonAlgorithmBase):
    input_data_type = ""
    t_port = 0
    t_index = 0
    model_path = ""
    class_path = ""

    def __init__(self):
        VTKPythonAlgorithmBase.__init__(
            self, nInputPorts=1, nOutputPorts=1, outputType="vtkRectilinearGrid")
        
    # First step in pipeline: set Port info
    def FillInputPortInformation(self, port, info):
        if port == 0:
            self.t_port = port
            self.t_index = info.Get(self.INPUT_CONNECTION())  # connection

        print("port info set")
        return 1

     # Request Data from input and create output
    def RequestData(self, request, inInfoVec, outInfoVec):
        from vtkmodules.vtkCommonDataModel import vtkRectilinearGrid, vtkImageData
        from vtkmodules.vtkCommonCore import VTK_DOUBLE

        self.input_data_type = self.GetInputDataObject(
            self.t_port, self.t_index).GetClassName()
        if self.input_data_type == "vtkRectilinearGrid":
            x, y, z, xCoords, yCoords, zCoords, vtk_output_array = self.Process_RectlinearGrid(
                inInfoVec, outInfoVec)

        output = vtkRectilinearGrid.GetData(outInfoVec, 0)
        output.SetDimensions(x, y, z)
        output.SetXCoordinates(xCoords)
        output.SetYCoordinates(yCoords)
        output.SetZCoordinates(zCoords)

        output.GetPointData().SetScalars(vtk_output_array)

        return 1

    def Process_RectlinearGrid(self, inInfoVec, outInfoVec):
        """
        Helper function to process info from input to send it for model inference
        Args:
        inInfoVec: input information vector
        outInfoVec: output information vector
        Returns:
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
        float_array = pdi.GetPointData().GetAbstractArray(0)
        no_components = float_array.GetNumberOfComponents()
        input_numpy_array = ns.vtk_to_numpy(float_array)

        output_np_array = self.Segment_Grid(input_numpy_array, no_components)

        output_vtk_array = DA.numpyTovtkDataArray(
            output_np_array, name="threshold_pixels")

        return x, y, z, xCoords, yCoords, zCoords, output_vtk_array

    def Segment_Grid(self, numpy_array, no_components):
        """
        Performs segmentation of grid according to model loaded.
        Args:
        numpy_array: Numpy array of magnitude of all cells of grid
        no_components: number of components of input VTK array
        Returns:
        segmented_np_array: numpy array with segmented labels for all cells
        """
        from importlib import import_module

        module_name = self.get_trained_class_module_name()
        module = import_module(module_name)

        net = module.Net()
        net.load_state_dict(torch.load(self.model_path))
        print(numpy_array.shape)

        torch_array = torch.from_numpy(numpy_array.copy()).to(torch.float)
        if no_components < 2:
            torch_array = torch.unsqueeze(torch_array, 1)
        print(torch_array.shape)

        outputs = net.predict(torch_array)
        segmented_np_array = outputs.detach().numpy()
        return segmented_np_array

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
