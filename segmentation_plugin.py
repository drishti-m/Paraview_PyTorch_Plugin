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
            rgb, x, y, z = self.Segment_Image(
                inInfoVec)

        output = vtkImageData.GetData(outInfoVec, 0)
        output.SetDimensions(x, y, 1)  # x, y, z)
        vtk_output = DA.numpyTovtkDataArray(rgb, name="numpy_array")
        print(vtk_output)
        output.GetPointData().SetScalars(vtk_output)

        return 1

    def Segment_Image(self, inInfoVec):
        from vtkmodules.vtkCommonDataModel import vtkRectilinearGrid, vtkImageData
        from vtkmodules.vtkCommonCore import VTK_DOUBLE
        pdi = vtkImageData.GetData(inInfoVec[0], 0)
        x, y, z = pdi.GetDimensions()
        print(x, y, z)
        # print(pdi.GetCellPoint(x, y))
        no_arrays = pdi.GetPointData().GetNumberOfArrays()
        float_array = pdi.GetPointData().GetAbstractArray(0)
        # print(self.GetPointData())
        # help(self)
        print(float_array)
        numpy_array = ns.vtk_to_numpy(float_array)
        print(numpy_array)

        numpy_array = numpy_array.reshape((y, x, z*3))  # , order="F")
        numpy_array = np.flip(numpy_array)
        print(numpy_array.shape)
        # print(numpy_array[0])

        fcn = torch.load(self.model_path)

        # numpy_array = numpy_array.resize(numpy_array, (256, 256, 3))

        trf = T.Compose([T.ToPILImage(), T.Resize(256),
                         T.ToTensor(), T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        inp = trf(numpy_array).unsqueeze(0)
        # print(inp)
        out = fcn(inp)['out']
        print('Calling model')
        om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
        rgb = self.decode_segmap(om)
        print(rgb.shape)
        r_x, r_y, r_z = rgb.shape
        # rgb.reshape((r_y,))
        rgb = np.flip(rgb)
        rgb = rgb.reshape((r_x*r_y, 3))  # , order="F")
        # print(rgb.shape)

        return rgb, r_y, r_x, r_z

    def decode_segmap(self, image, nc=21):

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

        # label_colors = np.array([(0, 0, 0),  # 0=background
        #                          # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
        #                          (10, 10, 10), (100, 100, 100), (120, 120,
        #                                                          120), (50, 50, 50), (150, 150, 150),
        #                          # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
        #                          (170, 170, 170), (128, 128, 128), (64,
        #                                                              64, 64), (192, 192, 192), (140, 140, 140),
        #                          # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
        #                          (160, 160, 160), (30, 30, 30), (250, 250,
        #                                                          250), (180, 180, 180), (200, 200, 200),
        #                          # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
        #                          (70, 70, 70), (20, 20, 20), (175, 175, 175), (135, 135, 135), (45, 45, 45)])

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

    @ smproperty.stringvector(name="Trained Model Path")
    def SetModelPathR(self, x):
        print("Model path: ", x)
        self.model_path = x
        self.Modified()
