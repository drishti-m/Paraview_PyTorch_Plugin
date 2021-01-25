# from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
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
from vtkmodules.numpy_interface import dataset_adapter as DA


@smproxy.filter()
@smproperty.input(name="InputImage", port_index=0)
@smdomain.datatype(dataTypes=["vtkImageData"], composite_data_supported=False)
# @smproperty.input(name="InputDataset", port_index=0)
# @smdomain.datatype(dataTypes=["vtkDataSet"], composite_data_supported=False)
class ThresholdImage(VTKPythonAlgorithmBase):
    threshold_cut = 0.1

    def __init__(self):
        VTKPythonAlgorithmBase.__init__(
            self, nInputPorts=1, nOutputPorts=1, outputType="vtkImageData")
        self.threshold_cut = 0.5

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
        print(self.threshold_cut)
        print("self:", self.threshold_cut)
        # PLUGIN different PART: pdi = vtkImagedata.GetData(inInfoVec[portNumber],0) instead of
        # prog filter: pdi = self.GetInput()
        pdi = vtkImageData.GetData(inInfoVec[0], 0)
        no_arrays = pdi.GetPointData().GetNumberOfArrays()
        float_array = pdi.GetPointData().GetAbstractArray(0)
        numpy_array = ns.vtk_to_numpy(float_array)
        numpy_array = numpy_array.reshape((3, 32, 32), order="F")
        true_false_array = numpy_array > self.threshold_cut
        predictions_np = true_false_array.astype(int)

        print(predictions_np.shape)
        two_dim_np_img = predictions_np.reshape(32, 32*3)
        vtk_float_array = DA.numpyTovtkDataArray(
            two_dim_np_img, name="numpy_array")
        print(vtk_float_array)

        # PLUGIN different part: output = GetData() instead of self.GetOutput()
        output = vtkImageData.GetData(outInfoVec, 0)
        output.SetDimensions(32, 32*3, 1)
        output.GetPointData().SetScalars(vtk_float_array)
        print(output)
        #dsa = output.GetRowData()
        # dsa.AddArray(predictions_vtk)

        print("Pretend work done!")
        return 1

    @ smproperty.xml("""
        <DoubleVectorProperty name="Threshold Pixel Value"
            number_of_elements="1"
            default_values="0"
            command="SetThreshold">
            <DoubleRangeDomain name="range" />
            <Documentation>Set threshold pixel value to segment(binary)</Documentation>
        </DoubleVectorProperty>""")
    def SetThreshold(self, x):
        # self._realAlgorithm.SetCenter(x, y, z)
        # self.Modified()
        # print(self)
        print("Threshold set", x)
        self.threshold_cut = x


@smproxy.filter()
@smproperty.input(name="InputRectilinear", port_index=0)
@smdomain.datatype(dataTypes=["vtkRectilinearGrid"], composite_data_supported=True)
@smproperty.output(name="OutputRectilinear", port_index=0)
@smdomain.datatype(dataTypes=["vtkRectilinearGrid"], composite_data_supported=True)
class ThresholdRectilinear(VTKPythonAlgorithmBase):
    threshold_cut = 0.5

    def __init__(self):
        self.threshold_cut = 0.5
        VTKPythonAlgorithmBase.__init__(
            self, nInputPorts=1, nOutputPorts=1, outputType="vtkRectilinearGrid")

    def FillInputPortInformation(self, port, info):
        #self = vtkPythonAlgorithm
        if port == 0:
            info.Set(self.INPUT_REQUIRED_DATA_TYPE(), "vtkRectilinearGrid")
            #info.Set(self.UPDATE_TIME_STEP(), 0)
            print("self:", self)
        print("port info set")
        return 1

    def RequestData(self, request, inInfoVec, outInfoVec):
        from vtkmodules.vtkCommonDataModel import vtkRectilinearGrid
        from vtkmodules.vtkCommonCore import VTK_DOUBLE

        '''inInfoVec = tuple
        inInfoVec[0] = vtkInformationVector
        outInfoVec = vtkInformationVector (not subscribtable)'''

        pdi = vtkRectilinearGrid.GetData(inInfoVec[0], 0)
        x, y, z = pdi.GetDimensions()
        xCoords = pdi.GetXCoordinates()
        yCoords = pdi.GetYCoordinates()
        zCoords = pdi.GetZCoordinates()
        no_arrays = pdi.GetPointData().GetNumberOfArrays()
        float_array = pdi.GetPointData().GetAbstractArray(0)
        numpy_array = ns.vtk_to_numpy(float_array)
        tf_array = numpy_array > self.threshold_cut
        seg_array = tf_array.astype(int)
        vtk_double_array = DA.numpyTovtkDataArray(
            seg_array, name="numpy_array")

        output = vtkRectilinearGrid.GetData(outInfoVec, 0)
        output.SetDimensions(x, y, z)
        output.SetXCoordinates(xCoords)
        output.SetYCoordinates(yCoords)
        output.SetZCoordinates(zCoords)
        output.GetPointData().SetScalars(vtk_double_array)
        print("output:", output)
        return 1

    @ smproperty.xml("""
        <DoubleVectorProperty name="Threshold Value"
            number_of_elements="1"
            default_values="0.5"
            command="SetThresholdR">
            <DoubleRangeDomain name="range" />
            <Documentation>Set threshold pixel value to segment(binary)</Documentation>
        </DoubleVectorProperty>""")
    def SetThresholdR(self, x):
        print("Threshold set", x)
        self.threshold_cut = x
        self.Modified()
