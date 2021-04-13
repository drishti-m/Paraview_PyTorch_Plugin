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
class ThresholdMaxImage(VTKPythonAlgorithmBase):
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
        # dsa = output.GetRowData()
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
@smdomain.datatype(dataTypes=["vtkRectilinearGrid", "vtkImageData"], composite_data_supported=True)
class ThresholdMaxRectilinear(VTKPythonAlgorithmBase):
    threshold_cut = 0.5
    input_data_type = ""
    t_port = 0
    t_index = 0
    path = ""

    def __init__(self):
        self.threshold_cut = 0.5
        VTKPythonAlgorithmBase.__init__(
            self, nInputPorts=1, nOutputPorts=1, outputType="vtkRectilinearGrid")

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
        from vtkmodules.vtkCommonDataModel import vtkRectilinearGrid, vtkImageData
        from vtkmodules.vtkCommonCore import VTK_DOUBLE

        '''inInfoVec = tuple
        inInfoVec[0] = vtkInformationVector
        outInfoVec = vtkInformationVector (not subscribtable)'''
        self.input_data_type = self.GetInputDataObject(
            self.t_port, self.t_index).GetClassName()
        if self.input_data_type == "vtkRectilinearGrid":
            x, y, z, xCoords, yCoords, zCoords, vtk_double_array = self.Process_RectlinearGrid(
                inInfoVec, outInfoVec)
        elif self.input_data_type == "vtkImageData":
            x, y, z, xCoords, yCoords, zCoords, vtk_double_array = self.Convert_Img_To_Rectilinear(
                inInfoVec)
        output = vtkRectilinearGrid.GetData(outInfoVec, 0)
        output.SetDimensions(x, y, z)
        output.SetXCoordinates(xCoords)
        output.SetYCoordinates(yCoords)
        output.SetZCoordinates(zCoords)
        output.GetPointData().SetScalars(vtk_double_array)

        return 1

    def Convert_Img_To_Rectilinear(self, inInfoVec):
        from vtkmodules.vtkCommonDataModel import vtkRectilinearGrid, vtkImageData
        from vtkmodules.vtkCommonCore import VTK_DOUBLE
        pdi = vtkImageData.GetData(inInfoVec[0], 0)
        x, y, z = pdi.GetDimensions()
        xCoords = np.arange(0, x, 1)
        yCoords = np.arange(0, y, 1)
        zCoords = np.arange(0, z, 1)
        xCoords = DA.numpyTovtkDataArray(
            xCoords, name="x-coordinates")
        yCoords = DA.numpyTovtkDataArray(
            yCoords, name="y-coordinates")
        zCoords = DA.numpyTovtkDataArray(
            zCoords, name="z-coordinates")

        no_arrays = pdi.GetPointData().GetNumberOfArrays()
        vtk_double_array = pdi.GetPointData().GetAbstractArray(0)
        numpy_array = ns.vtk_to_numpy(vtk_double_array)
        tf_array = numpy_array > self.threshold_cut
        seg_array = tf_array.astype(int)
        vtk_double_array = DA.numpyTovtkDataArray(
            seg_array, name="threshold_pixels")
        return x, y, z, xCoords, yCoords, zCoords, vtk_double_array

    def Process_RectlinearGrid(self, inInfoVec, outInfoVec):
        from vtkmodules.vtkCommonDataModel import vtkRectilinearGrid, vtkImageData
        from vtkmodules.vtkCommonCore import VTK_DOUBLE
        pdi = vtkRectilinearGrid.GetData(inInfoVec[0], 0)

        x, y, z = pdi.GetDimensions()
        xCoords = pdi.GetXCoordinates()
        yCoords = pdi.GetYCoordinates()
        zCoords = pdi.GetZCoordinates()
        no_arrays = pdi.GetPointData().GetNumberOfArrays()
        float_array = pdi.GetPointData().GetAbstractArray(0)
        numpy_array = ns.vtk_to_numpy(float_array)
        # print(numpy_array.shape)

        if float_array.GetNumberOfComponents() > 1:
            mag = np.array([np.sqrt(x.dot(x)) for x in numpy_array])
            mag = np.max(mag)
            tf_array = mag > self.threshold_cut
        else:
            tf_array = np.max(numpy_array)
            tf_array = tf_array > self.threshold_cut
        seg_array = np.full((x*y, z), tf_array[0].astype(int))
        # seg_array.fill(tf_array.astype(int))
        print(seg_array.shape, seg_array)
        vtk_double_array = DA.numpyTovtkDataArray(
            seg_array, name="numpy_array")

        output = vtkRectilinearGrid.GetData(outInfoVec, 0)

        return x, y, z, xCoords, yCoords, zCoords, vtk_double_array

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

    @ smproperty.stringvector(name="path")
    def SetPathR(self, x):
        print("Threshold set", x)
        self.path = x
        self.Modified()


@smproxy.filter()
@smproperty.input(name="InputRectilinear", port_index=0)
@smdomain.datatype(dataTypes=["vtkRectilinearGrid", "vtkImageData"], composite_data_supported=True)
class ThresholdMaxML(VTKPythonAlgorithmBase):
    input_data_type = ""
    t_port = 0
    t_index = 0
    model_path = ""
    class_path = ""

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
        from vtkmodules.vtkCommonDataModel import vtkRectilinearGrid, vtkImageData, vtkTable
        from vtkmodules.vtkCommonCore import VTK_DOUBLE

        self.input_data_type = self.GetInputDataObject(
            self.t_port, self.t_index).GetClassName()
        if self.input_data_type == "vtkRectilinearGrid":
            predictions_vtk, strArray, x, y, z, xCoords, yCoords, zCoords, vtk_double_array = self.Process_RectlinearGrid(
                inInfoVec, outInfoVec)
        elif self.input_data_type == "vtkImageData":
            x, y, z, xCoords, yCoords, zCoords, vtk_double_array = self.Convert_Img_To_Rectilinear(
                inInfoVec)

        out = vtkTable.GetData(outInfoVec, 0)
        dsa = out.GetRowData()
        dsa.AddArray(predictions_vtk)
        dsa.AddArray(strArray)
        # output = vtkRectilinearGrid.GetData(outInfoVec, 0)
        # output.SetDimensions(x, y, z)
        # output.SetXCoordinates(xCoords)
        # output.SetYCoordinates(yCoords)
        # output.SetZCoordinates(zCoords)

        # output.GetPointData().SetScalars(vtk_double_array)

        return 1

    def Convert_Img_To_Rectilinear(self, inInfoVec):
        from vtkmodules.vtkCommonDataModel import vtkRectilinearGrid, vtkImageData
        from vtkmodules.vtkCommonCore import VTK_DOUBLE
        pdi = vtkImageData.GetData(inInfoVec[0], 0)
        x, y, z = pdi.GetDimensions()
        xCoords = np.arange(0, x, 1)
        yCoords = np.arange(0, y, 1)
        zCoords = np.arange(0, z, 1)
        xCoords = DA.numpyTovtkDataArray(
            xCoords, name="x-coordinates")
        yCoords = DA.numpyTovtkDataArray(
            yCoords, name="y-coordinates")
        zCoords = DA.numpyTovtkDataArray(
            zCoords, name="z-coordinates")

        no_arrays = pdi.GetPointData().GetNumberOfArrays()
        vtk_double_array = pdi.GetPointData().GetAbstractArray(0)
        numpy_array = ns.vtk_to_numpy(vtk_double_array)

        torch_np_array = self.trained_threshold(numpy_array)

        vtk_double_array = DA.numpyTovtkDataArray(
            torch_np_array, name="threshold_pixels")
        return x, y, z, xCoords, yCoords, zCoords, vtk_double_array

    def Create_vtkTable_Columns(self, predictions_np, predicted_classes):
        predictions_vtk = ns.numpy_to_vtk(predictions_np)
        predictions_vtk.SetName("Predicted Values")
        strArray = vtk.vtkStringArray()

        strArray.SetName("Label names")
        strArray.SetNumberOfTuples(len(predicted_classes))
        for i in range(0, len(predicted_classes), 1):
            strArray.SetValue(i, predicted_classes[i])
        return predictions_vtk, strArray

    def Process_RectlinearGrid(self, inInfoVec, outInfoVec):
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
        numpy_array = ns.vtk_to_numpy(float_array)

        torch_np_array, predictions_np, predicted_classes = self.trained_threshold(
            numpy_array, no_components)
        predictions_vtk, strArray = self.Create_vtkTable_Columns(
            predictions_np, predicted_classes)

        print("Predicted: ", torch_np_array)
        torch_np_array = np.full((x*y, z), torch_np_array[0])
        vtk_double_array = DA.numpyTovtkDataArray(
            torch_np_array, name="threshold_pixels")
        # print(vtk_double_array)
        # vtk_double_array.SetNumberOfComponents(
        #     float_array.GetNumberOfComponents())

        return predictions_vtk, strArray, x, y, z, xCoords, yCoords, zCoords, vtk_double_array

    def trained_threshold(self, numpy_array, no_components):
        from importlib import import_module

        module_name = self.get_trained_class_module_name()
        #module_name = "neural_net"
        module = import_module(module_name)

        net = module.Net()
        net.load_state_dict(torch.load(self.model_path))
        # numpy_array = numpy_array.reshape((1, -1))[0]
        print(numpy_array.shape)

        torch_array = torch.from_numpy(numpy_array.copy()).to(torch.float)
        torch_array = torch.stack([torch_array, torch_array], dim=0)
        print(torch_array.shape)

        criterion = nn.CrossEntropyLoss()
        model_outputs = net.forward(torch_array)
        mag = np.array([np.sqrt(x.dot(x)) for x in torch_array])
        exp_output = np.max(mag) > 0.0
        exp_output = exp_output.astype(float)
        exp_output = np.array([exp_output, exp_output])
        exp_output = torch.from_numpy(exp_output).to(torch.long)
        v_loss = criterion(model_outputs, exp_output)
        timestep = self.GetInputDataObject(0, 0).GetInformation().Get(
            vtk.vtkDataObject.DATA_TIME_STEP())
        print("Loss in timestep", timestep, " = ", v_loss.item())

        classes = ["LOW", "HIGH"]
        o_tensors = net(torch_array)
        percentages = torch.nn.functional.softmax(o_tensors, dim=1)[0] * 100
        _, indices = torch.sort(o_tensors[0], descending=True)
        predictions_np = [percentages[idx].item()
                          for idx in indices]

        predicted_classes = [classes[idx]
                             for idx in indices]
        print(predictions_np, predicted_classes)

        outputs = net.predict(torch_array)
        # print(outputs)
        return outputs.detach().numpy(), predictions_np, predicted_classes

    def get_trained_class_module_name(self):
        import sys
        import os.path
        from pathlib import Path
        from importlib import import_module

        abs_path_class = os.path.abspath(self.class_path)
        class_dir = os.path.dirname(abs_path_class)
        sys.path.append(class_dir)
        module_name = Path(self.class_path).stem
        return module_name

    @ smproperty.stringvector(name="Trained Model Path")
    def SetModelPathR(self, x):
        print("Model path: ", x)
        self.model_path = x
        self.Modified()

    @ smproperty.stringvector(name="Model's Class Path")
    def SetClassPath(self, x):
        print("Class path: ", x)
        self.class_path = x
        self.Modified()
