import torch
import torchvision
from torchvision import models
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
import os
from vtk.util import numpy_support
from paraview.vtk.util import numpy_support as ns
from paraview import simple
from paraview.util.vtkAlgorithm import *
import numpy as np
from vtkmodules.numpy_interface import dataset_adapter as DA


@smproxy.filter()
@smproperty.input(name="segmentation", port_index=0)
@smdomain.datatype(dataTypes=["vtkImageData"], composite_data_supported=True)
class ML_Segmentation(VTKPythonAlgorithmBase):
    input_data_type = ""
    t_port = 0
    t_index = 0
    model_path = ""

    def __init__(self):
        VTKPythonAlgorithmBase.__init__(
            self, nInputPorts=1, nOutputPorts=1, outputType="vtkImageData")

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

        self.input_data_type = self.GetInputDataObject(
            self.t_port, self.t_index).GetClassName()
        # if self.input_data_type == "vtkRectilinearGrid":
        #     x, y, z, xCoords, yCoords, zCoords, vtk_double_array = self.Process_RectlinearGrid(
        #         inInfoVec, outInfoVec)

        if self.input_data_type == "vtkImageData":
            rgb_output_vtk, x, y, z = self.Segment_Image(
                inInfoVec)

        output = vtkImageData.GetData(outInfoVec, 0)
        output.SetDimensions(x, y, z)
        output.GetPointData().SetScalars(rgb_output_vtk)
        return 1

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
        pixels_np_array = np.flip(pixels_np_array)
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
        r_x, r_y, r_z = rgb_array.shape
        rgb_array = np.flip(rgb_array)
        rgb_array = rgb_array.reshape((r_x*r_y, r_z))
        vtk_output = DA.numpyTovtkDataArray(rgb_array, name="segmented_pixels")
        # x, y reversed between vtk <-> numpy array
        return vtk_output, r_y, r_x, r_z

    def Segment_Image(self, inInfoVec):
        """
        Performs pixel segmentation according to model loaded.

        Args:
        inInfovec: vtk Information vector containing input information from pipeline

        Returns:
        rgb_vtk: Segemnted pixel values array as suitable vtk array
        r_x: x-dimensions for resulting vtk image
        r_y: y-dimensions for resulting vtk image
        r_z: z-dimensions for resulting vtk image

        """
        from vtkmodules.vtkCommonDataModel import vtkRectilinearGrid, vtkImageData
        from vtkmodules.vtkCommonCore import VTK_DOUBLE
        pdi = vtkImageData.GetData(inInfoVec[0], 0)

        x, y, z = pdi.GetDimensions()
        print("Shape of input vtk Image: ", x, y, z)
        #no_arrays = pdi.GetPointData().GetNumberOfArrays()
        pixels_vtk_array = pdi.GetPointData().GetAbstractArray(0)
        # print(pixels_vtk_array)

        pixels_np_array = self.convert_vtk_to_numpy(pixels_vtk_array, x, y)
        print("Converted vtk array to suitable numpy representation with shape: ",
              pixels_np_array.shape)

        print("Loading model at path: ", self.model_path)
        fcn = torch.load(self.model_path)
        print("Preprocessing image for segmentation..")
        trf = T.Compose([T.ToPILImage(), T.Resize(256),
                         T.ToTensor(), T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        inp = trf(pixels_np_array).unsqueeze(0)
        out = fcn(inp)['out']

        print('\nResults ready. Check "segmented_pixels" coloring for results.')
        om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
        rgb = self.decode_segmap(om)

        rgb_vtk, r_x, r_y, r_z = self.convert_numpy_to_vtk(rgb)

        return rgb_vtk, r_x, r_y, r_z

    def decode_segmap(self, image, nc=21):
        """
        Decodes pixel colors for resulting values from semantic segmentation.

        Args: 
        image: image array containing label ids of segmentation
        nc = number of labels + 1

        Returns:
        rgb: rgb value of decoded pixels

        """
        label_colors = np.array([(0, 0, 0),  # 0=background
                                 # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                                 (128, 0, 0), (0, 128, 0), (128, 128,
                                                            0), (0, 0, 128), (128, 0, 128),
                                 # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                                 (0, 128, 128), (128, 128, 128), (64,
                                                                  0, 0), (192, 0, 0), (64, 128, 0),
                                 # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                                 (192, 128, 0), (64, 0, 128), (192, 0,
                                                               128), (64, 128, 128), (192, 128, 128),
                                 # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                                 (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

        r = np.zeros_like(image).astype(np.uint8)
        g = np.zeros_like(image).astype(np.uint8)
        b = np.zeros_like(image).astype(np.uint8)

        for l in range(0, nc):
            idx = image == l
            r[idx] = label_colors[l, 0]
            g[idx] = label_colors[l, 1]
            b[idx] = label_colors[l, 2]

        rgb = np.stack([r, g, b], axis=2)
        return rgb

    @smproperty.stringvector(name="Trained Model Path")
    def SetModelPathR(self, x):
        print("Model path: ", x)
        self.model_path = x
        self.Modified()