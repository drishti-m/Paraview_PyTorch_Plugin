from torchvision import models
import torch
import os

dir_name = os.path.dirname(os.path.realpath(__file__))
# download models for img segmentation
fcn = models.segmentation.fcn_resnet101(pretrained=True).eval()
path = dir_name + "/img-segment/fcn_resnet101.pth"
torch.save(fcn, path)
print("Saved at ", path)

dlab = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()
path = dir_name + "/img-segment/dlab_resnet101.pth"
torch.save(dlab, path)
print("Saved at ", path)

# download models for img classification
alexnet = models.alexnet(pretrained=True).eval()
path = dir_name + "/img-classify/alexnet.pth"
torch.save(alexnet, path)
print("Saved at ", path)

resnet = models.resnet101(pretrained=True).eval()
path = dir_name + "/img-classify/resnet.pth"
torch.save(resnet, path)
print("Saved at ", path)
