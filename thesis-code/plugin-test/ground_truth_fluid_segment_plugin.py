"""
This is the plugin code for threshold segmentation(binary) of rectilinear grid. 
This plugin is designed to segment a grid such that cells with magnitude greater than the threshold value are
labelled as 1, otherwise labelled as 0. This provides the ground truth results for
such a fluid segmentation.
Accepted input data models: VTK Rectilinear Grid
Output data model: VTK Rectilinear Grid
It takes from user two parameters: Threshold value.

"""

from paraview.util.vtkAlgorithm import *
from paraview.vtk.util import numpy_support as ns
from paraview import simple
import vtk
from vtk.util import numpy_support
from vtkmodules.numpy_interface import dataset_adapter as DA
import numpy as np


@smproxy.filter()
@smproperty.input(name="InputRectilinear", port_index=0)
@smdomain.datatype(dataTypes=["vtkRectilinearGrid"], composite_data_supported=True)
class ThresholdRectilinear(VTKPythonAlgorithmBase):
    threshold_cut = 0.01
    input_data_type = ""
    t_port = 0
    t_index = 0

    def __init__(self):
        self.threshold_cut = 0.5
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

        '''inInfoVec = tuple
        inInfoVec[0] = vtkInformationVector
        outInfoVec = vtkInformationVector (not subscribtable)'''
        self.input_data_type = self.GetInputDataObject(
            self.t_port, self.t_index).GetClassName()
        if self.input_data_type == "vtkRectilinearGrid":
            x, y, z, xCoords, yCoords, zCoords, vtk_double_array = self.Process_RectlinearGrid(
                inInfoVec, outInfoVec)

        output = vtkRectilinearGrid.GetData(outInfoVec, 0)
        output.SetDimensions(x, y, z)
        output.SetXCoordinates(xCoords)
        output.SetYCoordinates(yCoords)
        output.SetZCoordinates(zCoords)
        output.GetPointData().SetScalars(vtk_double_array)

        return 1


    def Process_RectlinearGrid(self, inInfoVec, outInfoVec):     
       """
        Helper function to process info from input and carry out threshold segmentation.
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
        numpy_array = ns.vtk_to_numpy(float_array)
        # print(numpy_array.shape)

        if float_array.GetNumberOfComponents() > 1:
            mag = np.array([np.sqrt(x.dot(x)) for x in numpy_array])
            tf_array = mag > self.threshold_cut
        else:
            tf_array = numpy_array > self.threshold_cut
        seg_array = tf_array.astype(int)
        output_vtk_array = DA.numpyTovtkDataArray(
            seg_array, name="numpy_array")

        output = vtkRectilinearGrid.GetData(outInfoVec, 0)

        return x, y, z, xCoords, yCoords, zCoords, output_vtk_array

    @ smproperty.xml("""
        <DoubleVectorProperty name="Threshold Value"
            number_of_elements="1"
            default_values="0.01"
            command="SetThresholdR">
            <DoubleRangeDomain name="range" />
            <Documentation>Set threshold pixel value to segment(binary)</Documentation>
        </DoubleVectorProperty>""")
    def SetThresholdR(self, x):
        print("Threshold set", x)
        self.threshold_cut = x
        self.Modified()


