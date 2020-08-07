from torchvision import models

import torch.nn as nn
import torch.nn.functional as F


class DogClassificationModel(nn.Module):
    def __init__(self, model, num_classes, mean=0.5, std=0.25):
        super(DogClassificationModel, self).__init__()
        self.num_classes = num_classes
        self.mean = mean
        self.std = std

        # base model (e.g. ResNet50, ResNet152, ResNeXt, ... and so on.)
        self.base_model = model
        num_ftrs = model.fc.in_features
        self.base_model.fc = nn.Linear(num_ftrs, self.num_classes)

    def forward(self, x):
        x = (x - self.mean) / self.std 
        x = self.base_model(x)
        return x


class OrgModel(nn.Module):
    def __init__(self, num_classes, attention=False):
        super(OrgModel, self).__init__()
        self.attention = attention

        self.base_model = models.resnet50(pretrained=True)
        layers = list(self.base_model.children())[:4]
        self.model0 = nn.Sequential(*layers)
        self.model1 = list(self.base_model.children())[4]

        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)

        layers2 = list(self.base_model.children())[5:9] 
        self.model2 = nn.Sequential(*layers2)

        self.fc = nn.Linear(2048, num_classes)


    def forward(self, x):
        x = (x - 0.5) / 0.25
        x = self.model0(x)
        x = self.model1(x)
        y = F.adaptive_max_pool2d(x, (1, 1)).squeeze(2).squeeze(2)
        y = self.fc1(y)
        y = self.fc2(y)
        y = y.view(-1, 256, 1, 1)

        z = x
        if self.attention:
            z = x * y
        x = self.model2(z)
        x = x.view(-1, 2048)
        x = self.fc(x)

        return x, z 


class ExtendedModelForCoreML(nn.Module):
    """
    The model with the additional layer to convert pixel ragne [0, 255] to [0, 1] for CoreML.
    """
    def __init__(self, model):
        super(ExtendedModelForCoreML, self).__init__()
        self.model = model  # model: DogClassificationModel


    def forward(self, x):
        x = x / 255.0
        return self.model(x)