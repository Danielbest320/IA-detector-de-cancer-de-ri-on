# model.py
from torchvision.models import resnet18
import torch.nn as nn

def get_modified_resnet18(num_classes=2):
    model = resnet18(pretrained=True)
    model.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
