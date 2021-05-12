from torchvision import models
import torch

# download models for img segmentation
fcn = models.segmentation.fcn_resnet101(pretrained=True).eval()
torch.save(fcn, "./img-segment/fcn_resnet101.pth")
print("Saved at ./pre-trained-models/img-segment/fcn_resnet101.pth")

dlab = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()
torch.save(dlab, "./img-segment/dlab_resnet101.pth")
print("Saved at ./pre-trained-models/img-segment/dlab_resnet101.pth")

# download models for img classification
alexnet = models.alexnet(pretrained=True).eval()
torch.save(alexnet, "./img-classify/alexnet.pth")
print("Saved at ./pre-trained-models/img-classify/alexnet.pth")

resnet = models.resnet101(pretrained=True).eval()
torch.save(resnet, "./img-classify/resnet.pth")
print("Saved at ./pre-trained-models/img-classify/resnet.pth")
