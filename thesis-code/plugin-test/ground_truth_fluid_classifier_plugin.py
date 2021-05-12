"""
This is the plugin code for classification(binary) of rectilinear grid. 
This plugin is designed to classify a grid as "High" if max value of grid
is greater than threshold value, otherwise "Low".
Accepted input data models: VTK Rectilinear Grid, VTK Image Data
Output data model: VTK Table
It takes from user one parameter: Threshold value.

"""

from paraview.util.vtkAlgorithm import *
from paraview.vtk.util import numpy_support as ns
import vtk
import numpy as np


@smproxy.filter()
@smproperty.input(name="InputRectilinear", port_index=0)
@smdomain.datatype(dataTypes=["vtkRectilinearGrid", "vtkImageData"], composite_data_supported=True)
class ThresholdMaxRectilinear(VTKPythonAlgorithmBase):
    threshold_cut = 0.0
    input_data_type = ""
    t_port = 0
    t_index = 0
    path = ""

    def __init__(self):
        self.threshold_cut = 0.5
        VTKPythonAlgorithmBase.__init__(
            self, nInputPorts=1, nOutputPorts=1, outputType="vtkTable")

    # First step in pipeline: set Port info
    def FillInputPortInformation(self, port, info):
        if port == 0:
            self.t_port = port
            self.t_index = info.Get(self.INPUT_CONNECTION())  # connection

        print("port info set")
        return 1

    # Request Data from input and create output
    def RequestData(self, request, inInfoVec, outInfoVec):
        from vtkmodules.vtkCommonDataModel import vtkRectilinearGrid, vtkImageData, vtkTable
        from vtkmodules.vtkCommonCore import VTK_DOUBLE

        '''inInfoVec = tuple
        inInfoVec[0] = vtkInformationVector
        outInfoVec = vtkInformationVector (not subscribtable)'''
        self.input_data_type = self.GetInputDataObject(
            self.t_port, self.t_index).GetClassName()
        if self.input_data_type == "vtkRectilinearGrid":
            strArray = self.Process_RectlinearGrid(
                inInfoVec)
        elif self.input_data_type == "vtkImageData":
            strArray = self.Convert_Img_To_Rectilinear(
                inInfoVec)

        out = vtkTable.GetData(outInfoVec, 0)
        dsa = out.GetRowData()
        dsa.AddArray(strArray)

        return 1

    def Convert_Img_To_Rectilinear(self, inInfoVec):
        """
        Helper function to process image data same as rectilinear grid, and to provide
        output the ground truth label for classification.

        Args: 
        inInfovec: input info vector

        Returns:
        strArray: field for Table Data with output array

        """
        from vtkmodules.vtkCommonDataModel import vtkRectilinearGrid, vtkImageData
        from vtkmodules.vtkCommonCore import VTK_DOUBLE
        pdi = vtkImageData.GetData(inInfoVec[0], 0)
        no_arrays = pdi.GetPointData().GetNumberOfArrays()
        vtk_double_array = pdi.GetPointData().GetAbstractArray(0)
        numpy_array = ns.vtk_to_numpy(vtk_double_array)
        tf_array = numpy_array > self.threshold_cut
        print("Max value of pixel greater than ",
              self.threshold_cut, "?: ", tf_array)
        strArray = self.Create_vtkTable_Columns(tf_array)
        return strArray

    def Process_RectlinearGrid(self, inInfoVec):
        """
        Helper function to process rectilinear grid, and to provide
        output the ground truth label for classification.

        Args: 
        inInfovec: input info vector

        Returns:
        strArray: field for Table Data with output array

        """
        from vtkmodules.vtkCommonDataModel import vtkRectilinearGrid
        pdi = vtkRectilinearGrid.GetData(inInfoVec[0], 0)
        float_array = pdi.GetPointData().GetAbstractArray(0)
        numpy_array = ns.vtk_to_numpy(float_array)

        if float_array.GetNumberOfComponents() > 1:
            mag = np.array([np.sqrt(x.dot(x)) for x in numpy_array])
            mag_max = np.max(mag)
            tf_array = mag_max > self.threshold_cut
        else:
            tf_array = np.max(numpy_array)
            tf_array = tf_array > self.threshold_cut

        print("Max value of grid greater than ",
              self.threshold_cut, "?: ", tf_array)
        strArray = self.Create_vtkTable_Columns(tf_array)

        return strArray

    def Create_vtkTable_Columns(self, label_bool):
        """
        Helper function to create rows and columns for table

        Args:
        label_bool: bool value indicating if max > threshold

        Returns:
        strArray: VTK array of ground truth label
        """
        if label_bool == True:
            label = "High"
        else:
            label = "Low"
        strArray = vtk.vtkStringArray()
        strArray.SetName("Ground truth Label")
        strArray.SetNumberOfTuples(1)
        strArray.SetValue(0, "High")
        return strArray

    @ smproperty.xml("""
        <DoubleVectorProperty name="Threshold Value"
            number_of_elements="1"
            default_values="0.0"
            command="SetThresholdR">
            <DoubleRangeDomain name="range" />
            <Documentation>Set threshold pixel value to segment(binary)</Documentation>
        </DoubleVectorProperty>""")
    def SetThresholdR(self, x):
        print("Threshold set", x)
        self.threshold_cut = x
        self.Modified()
