from torchvision import models
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(Attention, self).__init__()

        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_channels, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        x = F.adaptive_max_pool2d(x, (1, 1))
        x = x.view(-1, x.shape[1])
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = x.view(-1, x.shape[1], 1, 1)
        return x * identity


class AttentionResnet50(nn.Module):
    def __init__(self, num_classes, use_attention=False, pretrained=False):
        super(AttentionResnet50, self).__init__()

        # num of classes
        self.num_classes = num_classes

        # attention?
        self.use_attention = use_attention

        # base model
        base_model = models.resnet50(pretrained=pretrained)

        # all layers
        layers = list(base_model.children())

        # Downsample: 1, 2, 3, 4
        self.prelayers = nn.Sequential(*(layers[:4]))   # 0, 1, 2, 3

        # Resnet1: 5
        resnet1_layers = list(layers[4].children())

        #   Bottleneck: 5-1
        resnet1_bottleneck1_layers = list(resnet1_layers[0].children())
        self.resnet1_bottleneck1_pre = nn.Sequential(*(resnet1_bottleneck1_layers[:5])) # Conv2d > BN > Conv2d > BN > Conv2d > BN
        self.resnet1_bottleneck1_skip = nn.Sequential(*(resnet1_bottleneck1_layers[7])) # Sequntial(Conv2d > BN)
        self.resnet1_bottleneck1_relu = resnet1_bottleneck1_layers[6]
        self.attention1_1 = Attention(256, 64)

        #   Bottleneck: 5-2
        resnet1_bottleneck2_layers = list(resnet1_layers[1].children())
        self.resnet1_bottleneck2_pre = nn.Sequential(*(resnet1_bottleneck2_layers[:5]))
        self.resnet1_bottleneck2_relu = resnet1_bottleneck2_layers[6] 
        self.attention1_2 = Attention(256, 64)

        #   Bottleneck: 5-3
        resnet1_bottleneck3_layers = list(resnet1_layers[2].children())
        self.resnet1_bottleneck3_pre = nn.Sequential(*(resnet1_bottleneck3_layers[:5]))
        self.resnet1_bottleneck3_relu = resnet1_bottleneck3_layers[6] 
        self.attention1_3 = Attention(256, 64)

        # post layers
        self.post_layers = nn.Sequential(*(layers[5:9]))
        self.fc = nn.Linear(2048, self.num_classes) 

        
    def forward(self, x):
        # normalizing inputs
        x = (x - 0.5) / 0.25 

        # downsample
        x = self.prelayers(x)

        # Resnet1
        #   Bottleneck1
        y = self.resnet1_bottleneck1_pre(x)
        z = self.resnet1_bottleneck1_skip(x)
        y = self.attention1_1(y)
        x = self.resnet1_bottleneck1_relu(y + z)

        #   Bottleneck2
        y = self.resnet1_bottleneck2_pre(x)
        y = self.attention1_2(y)
        x = self.resnet1_bottleneck2_relu(x + y)
        #print(x.shape)

        #   Bottleneck3
        y = self.resnet1_bottleneck3_pre(x)
        y = self.attention1_3(y)
        x = self.resnet1_bottleneck3_relu(x + y)
        #print(x.shape)

        x = self.post_layers(x)
        x = x.view(-1, 2048)

        return self.fc(x)

if __name__ == '__main__':
    model = AttentionResnet50(num_classes=120, use_attention=True, pretrained=True)

    image = np.random.rand(1, 3, 224, 224)
    tensor = torch.from_numpy(image).float()

    outputs = model(tensor)

    print("----- result ------")
    print(outputs.shape)
