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

        # relu
        self.relu = nn.ReLU(inplace=True)

        self.prelayers = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool)

        layer1 = base_model.layer1
        layer2 = base_model.layer2
        layer3 = base_model.layer3
        layer4 = base_model.layer4

        # Resnet1
        bottleneck1_1 = layer1[0]
        self.bottleneck1_1 = nn.Sequential(
            bottleneck1_1.conv1,
            bottleneck1_1.bn1, 
            self.relu,
            bottleneck1_1.conv2,
            bottleneck1_1.bn2,
            self.relu,
            bottleneck1_1.conv3,
            bottleneck1_1.bn3)
        self.resnet1_downsample = bottleneck1_1.downsample
        
        bottleneck1_2 = layer1[1]
        self.bottleneck1_2 = nn.Sequential(
            bottleneck1_2.conv1,
            bottleneck1_2.bn1, 
            self.relu,
            bottleneck1_2.conv2,
            bottleneck1_2.bn2,
            self.relu,
            bottleneck1_2.conv3,
            bottleneck1_2.bn3)

        bottleneck1_3 = layer1[2]
        self.bottleneck1_3 = nn.Sequential(
            bottleneck1_3.conv1,
            bottleneck1_3.bn1, 
            self.relu,
            bottleneck1_3.conv2,
            bottleneck1_3.bn2,
            self.relu,
            bottleneck1_3.conv3,
            bottleneck1_3.bn3)

        # Resnet2
        bottleneck2_1 = layer2[0]
        self.bottleneck2_1 = nn.Sequential(
            bottleneck2_1.conv1,
            bottleneck2_1.bn1, 
            self.relu,
            bottleneck2_1.conv2,
            bottleneck2_1.bn2,
            self.relu,
            bottleneck2_1.conv3,
            bottleneck2_1.bn3)
        self.resnet2_downsample = bottleneck2_1.downsample
        
        bottleneck2_2 = layer2[1]
        self.bottleneck2_2 = nn.Sequential(
            bottleneck2_2.conv1,
            bottleneck2_2.bn1, 
            self.relu,
            bottleneck2_2.conv2,
            bottleneck2_2.bn2,
            self.relu,
            bottleneck2_2.conv3,
            bottleneck2_2.bn3)

        bottleneck2_3 = layer2[2]
        self.bottleneck2_3 = nn.Sequential(
            bottleneck2_3.conv1,
            bottleneck2_3.bn1, 
            self.relu,
            bottleneck2_3.conv2,
            bottleneck2_3.bn2,
            self.relu,
            bottleneck2_3.conv3,
            bottleneck2_3.bn3)

        bottleneck2_4 = layer2[3]
        self.bottleneck2_4 = nn.Sequential(
            bottleneck2_4.conv1,
            bottleneck2_4.bn1, 
            self.relu,
            bottleneck2_4.conv2,
            bottleneck2_4.bn2,
            self.relu,
            bottleneck2_4.conv3,
            bottleneck2_4.bn3)

        # Resnet3
        bottleneck3_1 = layer3[0]
        self.bottleneck3_1 = nn.Sequential(
            bottleneck3_1.conv1,
            bottleneck3_1.bn1, 
            self.relu,
            bottleneck3_1.conv2,
            bottleneck3_1.bn2,
            self.relu,
            bottleneck3_1.conv3,
            bottleneck3_1.bn3)
        self.resnet3_downsample = bottleneck3_1.downsample
        
        bottleneck3_2 = layer3[1]
        self.bottleneck3_2 = nn.Sequential(
            bottleneck3_2.conv1,
            bottleneck3_2.bn1, 
            self.relu,
            bottleneck3_2.conv2,
            bottleneck3_2.bn2,
            self.relu,
            bottleneck3_2.conv3,
            bottleneck3_2.bn3)

        bottleneck3_3 = layer3[2]
        self.bottleneck3_3 = nn.Sequential(
            bottleneck3_3.conv1,
            bottleneck3_3.bn1, 
            self.relu,
            bottleneck3_3.conv2,
            bottleneck3_3.bn2,
            self.relu,
            bottleneck3_3.conv3,
            bottleneck3_3.bn3)

        bottleneck3_4 = layer3[3]
        self.bottleneck3_4 = nn.Sequential(
            bottleneck3_4.conv1,
            bottleneck3_4.bn1, 
            self.relu,
            bottleneck3_4.conv2,
            bottleneck3_4.bn2,
            self.relu,
            bottleneck3_4.conv3,
            bottleneck3_4.bn3)

        bottleneck3_5 = layer3[4]
        self.bottleneck3_5 = nn.Sequential(
            bottleneck3_5.conv1,
            bottleneck3_5.bn1, 
            self.relu,
            bottleneck3_5.conv2,
            bottleneck3_5.bn2,
            self.relu,
            bottleneck3_5.conv3,
            bottleneck3_5.bn3)

        bottleneck3_6 = layer3[5]
        self.bottleneck3_6 = nn.Sequential(
            bottleneck3_6.conv1,
            bottleneck3_6.bn1, 
            self.relu,
            bottleneck3_6.conv2,
            bottleneck3_6.bn2,
            self.relu,
            bottleneck3_6.conv3,
            bottleneck3_6.bn3)

        # Resnet4
        bottleneck4_1 = layer4[0]
        self.bottleneck4_1 = nn.Sequential(
            bottleneck4_1.conv1,
            bottleneck4_1.bn1, 
            self.relu,
            bottleneck4_1.conv2,
            bottleneck4_1.bn2,
            self.relu,
            bottleneck4_1.conv3,
            bottleneck4_1.bn3)
        self.resnet4_downsample = bottleneck4_1.downsample
        
        bottleneck4_2 = layer4[1]
        self.bottleneck4_2 = nn.Sequential(
            bottleneck4_2.conv1,
            bottleneck4_2.bn1, 
            self.relu,
            bottleneck4_2.conv2,
            bottleneck4_2.bn2,
            self.relu,
            bottleneck4_2.conv3,
            bottleneck4_2.bn3)

        bottleneck4_3 = layer4[2]
        self.bottleneck4_3 = nn.Sequential(
            bottleneck4_3.conv1,
            bottleneck4_3.bn1, 
            self.relu,
            bottleneck4_3.conv2,
            bottleneck4_3.bn2,
            self.relu,
            bottleneck4_3.conv3,
            bottleneck4_3.bn3)

        self.avgpool = base_model.avgpool
        self.fc = nn.Linear(2048, num_classes)

        
    def forward(self, x):
        # normalizing inputs
        x = (x - 0.5) / 0.25 

        # downsample
        x = self.prelayers(x)

        #
        # Resnet1
        #
        y = self.resnet1_downsample(x)
        x = self.bottleneck1_1(x)
        x = self.relu(x + y)

        identity = x
        y = self.bottleneck1_2(x)
        x = self.relu(y + identity)

        identity = x
        y = self.bottleneck1_3(x)
        x = self.relu(y + identity)

        #
        # ResNet2
        #
        y = self.bottleneck2_1(x)
        x = self.resnet2_downsample(x)
        x = self.relu(y + x)

        identity = x
        y = self.bottleneck2_2(x)
        x = self.relu(y + identity)

        identity = x
        y = self.bottleneck2_3(x)
        x = self.relu(y + identity)

        identity = x
        y = self.bottleneck2_4(x)
        x = self.relu(y + identity)

        #
        # ResNet3
        #
        y = self.bottleneck3_1(x)
        x = self.resnet3_downsample(x)
        x = self.relu(x + y)

        identity = x
        y = self.bottleneck3_2(x)
        x = self.relu(y + identity)

        identity = x
        y = self.bottleneck3_3(x)
        x = self.relu(y + identity)

        identity = x
        y = self.bottleneck3_4(x)
        x = self.relu(y + identity)

        identity = x
        y = self.bottleneck3_5(x)
        x = self.relu(y + identity)

        identity = x
        y = self.bottleneck3_6(x)
        x = self.relu(y + identity)

        #
        # ResNet4
        #
        y = self.bottleneck4_1(x)
        x = self.resnet4_downsample(x)
        x = self.relu(x + y)

        identity = x
        y = self.bottleneck4_2(x)
        x = self.relu(y + identity)

        identity = x
        y = self.bottleneck4_3(x)
        x = self.relu(y + identity)

        # post process
        x = self.avgpool(x)
        x = x.view(-1, 2048)

        return self.fc(x)


class AttentionResnet50Sub(nn.Module):
    def __init__(self, num_classes, use_attention=False, pretrained=False):
        super(AttentionResnet50Sub, self).__init__()

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
        self.resnet1 = layers[4]
        self.attention1 = Attention(256, 256//4)

        # Resnet2: 6
        self.resnet2 = layers[5]
        self.attention2 = Attention(512, 512//4)

        # Resnet3: 7
        self.resnet3 = layers[6]
        self.attention3 = Attention(1024, 1024//4)

        # Resnet4: 8
        self.resnet4 = layers[7]
        self.attention4 = Attention(2048, 2048//4)

        # post layers
        self.post_layers = nn.Sequential(*(layers[8:9]))
        self.fc = nn.Linear(2048, self.num_classes) 

        
    def forward(self, x):
        # normalizing inputs
        x = (x - 0.5) / 0.25 

        # downsample
        x = self.prelayers(x)

        # Resnet1
        x = self.resnet1(x)
        if self.use_attention:
            x = self.attention1(x)
 
        # Resnet2
        x = self.resnet2(x)
        if self.use_attention:
            x = self.attention2(x)

        # Resnet3
        x = self.resnet3(x)
        if self.use_attention:
            x = self.attention3(x)

        # Resnet4
        x = self.resnet4(x)
        if self.use_attention:
            x = self.attention4(x)

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
