from torchvision import models

import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    def __init__(self, in_channels):
        super(Bottleneck, self).__init__()
        self.block1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(in_channels), nn.ReLU())
        self.block2 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(in_channels), nn.ReLU())
        self.relu = nn.ReLU()

    def forward(self, x):
        id = x
        x = self.block1(x)
        x = self.block2(x)
        x = x + id

        return self.relu(x)


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()

        self.down1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.resid1 = nn.Sequential(
            Bottleneck(in_channels=64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            Bottleneck(in_channels=64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            Bottleneck(in_channels=64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.resid2 = nn.Sequential(
            Bottleneck(in_channels=128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            Bottleneck(in_channels=128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.resid3 = nn.Sequential(
            Bottleneck(in_channels=256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            Bottleneck(in_channels=256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            Bottleneck(in_channels=256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            Bottleneck(in_channels=256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            Bottleneck(in_channels=256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.down4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.resid4 = nn.Sequential(
            Bottleneck(in_channels=512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            Bottleneck(in_channels=512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 121)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.down1(x)
        x = self.resid1(x)
        x = self.down2(x)
        x = self.resid2(x)
        x = self.down3(x)
        x = self.resid3(x)
        x = self.down4(x)
        x = self.resid4(x)
        x = F.adaptive_max_pool2d(x, (1, 1))

        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x, x


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

        self.fc1 = nn.Linear(256, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 256)
        self.sigmoid = nn.Sigmoid()

        layers2 = list(self.base_model.children())[5:9] 
        self.model2 = nn.Sequential(*layers2)

        self.fc = nn.Linear(2048, num_classes)


    def forward(self, x):
        x = (x - 0.5) / 0.25
        x = self.model0(x)
        x = self.model1(x)
        y = F.adaptive_max_pool2d(x, (1, 1)).squeeze(2).squeeze(2)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
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