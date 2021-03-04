# Prg filter
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
import numpy as np
from vtkmodules.numpy_interface import dataset_adapter as DA


pdi = self.GetInput()
x, y, z = pdi.GetDimensions()
print(x, y, z)
#print(pdi.GetCellPoint(x, y))
no_arrays = pdi.GetPointData().GetNumberOfArrays()
float_array = pdi.GetPointData().GetAbstractArray(0)
# print(self.GetPointData())
# help(self)
print(float_array)
numpy_array = ns.vtk_to_numpy(float_array)
print(numpy_array.shape)
numpy_array = numpy_array.reshape((x, y, z*3), order="F")
#print("NUMPY ARAY")
# print(numpy_array[1])
print(os.getcwd())
PATH = './pytorch_data/seg_net.pth'

#fcn = models.segmentation.fcn_resnet101(pretrained=True).eval()
fcn = torch.load(PATH)
print("Loaded pre-trained model 'seg_net.pth'..")


def decode_segmap(image, nc=21):

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


#torch_array = torch.from_numpy(numpy_array)
# print(torch_array)
trf = T.Compose([T.ToTensor(), T.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
inp = trf(numpy_array).unsqueeze(0)
# print(inp)
out = fcn(inp)['out']
print('Calling model')
om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
rgb = decode_segmap(om)
print(rgb.shape)

out = self.GetOutput()
out.SetDimensions(x, y, z)
rgb = rgb.reshape((x*y, 3), order="F")

vtk_output = DA.numpyTovtkDataArray(rgb, name="numpy_array")
out.GetPointData().SetScalars(vtk_output)
