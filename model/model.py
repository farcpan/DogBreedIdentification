import torch.nn as nn


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