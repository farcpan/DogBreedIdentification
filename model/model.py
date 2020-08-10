from torchvision import models

import torch.nn as nn
import torch.nn.functional as F


class DogClassificationModel(nn.Module):
    def __init__(self, model, num_classes, batch_norm=False):
        super(DogClassificationModel, self).__init__()
        self.num_classes = num_classes
        self.batch_norm = batch_norm

        # base model (e.g. ResNet50, ResNet152, ResNeXt, ... and so on.)
        self.base_model = model
        num_ftrs = model.fc.in_features
        self.base_model.fc = nn.Linear(num_ftrs, self.num_classes)


    def forward(self, x):
        mean = 0.5
        std = 0.25
        if self.batch_norm:
            mean = torch.mean(x)
            std = torch.std(x)

        x = (x - mean) / std 
        x = self.base_model(x)
        return x


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
