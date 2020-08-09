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
        self.attention1_1 = Attention(256, 256//16)

        #   Bottleneck: 5-2
        resnet1_bottleneck2_layers = list(resnet1_layers[1].children())
        self.resnet1_bottleneck2_pre = nn.Sequential(*(resnet1_bottleneck2_layers[:5]))
        self.resnet1_bottleneck2_relu = resnet1_bottleneck2_layers[6] 
        self.attention1_2 = Attention(256, 256//16)

        #   Bottleneck: 5-3
        resnet1_bottleneck3_layers = list(resnet1_layers[2].children())
        self.resnet1_bottleneck3_pre = nn.Sequential(*(resnet1_bottleneck3_layers[:5]))
        self.resnet1_bottleneck3_relu = resnet1_bottleneck3_layers[6] 
        self.attention1_3 = Attention(256, 256//16)

        # Resnet2: 6
        resnet2_layers = list(layers[5].children())

        #   Bottleneck: 6-1
        resnet2_bottleneck1_layers = list(resnet2_layers[0].children())
        self.resnet2_bottleneck1_pre = nn.Sequential(*(resnet2_bottleneck1_layers[:5])) # Conv2d > BN > Conv2d > BN > Conv2d > BN
        self.resnet2_bottleneck1_skip = nn.Sequential(*(resnet2_bottleneck1_layers[7])) # Sequntial(Conv2d > BN)
        self.resnet2_bottleneck1_relu = resnet2_bottleneck1_layers[6]
        self.attention2_1 = Attention(512, 512//16)

        #   Bottleneck: 6-2
        resnet2_bottleneck2_layers = list(resnet2_layers[1].children())
        self.resnet2_bottleneck2_pre = nn.Sequential(*(resnet2_bottleneck2_layers[:5]))
        self.resnet2_bottleneck2_relu = resnet2_bottleneck2_layers[6] 
        self.attention2_2 = Attention(512, 512//16)

        #   Bottleneck: 6-3
        resnet2_bottleneck3_layers = list(resnet2_layers[2].children())
        self.resnet2_bottleneck3_pre = nn.Sequential(*(resnet2_bottleneck3_layers[:5]))
        self.resnet2_bottleneck3_relu = resnet2_bottleneck3_layers[6] 
        self.attention2_3 = Attention(512, 512//16)

        #   Bottleneck: 6-4
        resnet2_bottleneck4_layers = list(resnet2_layers[3].children())
        self.resnet2_bottleneck4_pre = nn.Sequential(*(resnet2_bottleneck4_layers[:5]))
        self.resnet2_bottleneck4_relu = resnet2_bottleneck4_layers[6] 
        self.attention2_4 = Attention(512, 512//16)

        # Resnet3: 7
        resnet3_layers = list(layers[6].children())

        #   Bottleneck: 7-1
        resnet3_bottleneck1_layers = list(resnet3_layers[0].children())
        self.resnet3_bottleneck1_pre = nn.Sequential(*(resnet3_bottleneck1_layers[:5])) # Conv2d > BN > Conv2d > BN > Conv2d > BN
        self.resnet3_bottleneck1_skip = nn.Sequential(*(resnet3_bottleneck1_layers[7])) # Sequntial(Conv2d > BN)
        self.resnet3_bottleneck1_relu = resnet3_bottleneck1_layers[6]
        self.attention3_1 = Attention(1024, 1024//16)

        #   Bottleneck: 7-2
        resnet3_bottleneck2_layers = list(resnet3_layers[1].children())
        self.resnet3_bottleneck2_pre = nn.Sequential(*(resnet3_bottleneck2_layers[:5]))
        self.resnet3_bottleneck2_relu = resnet3_bottleneck2_layers[6] 
        self.attention3_2 = Attention(1024, 1024//16)

        #   Bottleneck: 7-3
        resnet3_bottleneck3_layers = list(resnet3_layers[2].children())
        self.resnet3_bottleneck3_pre = nn.Sequential(*(resnet3_bottleneck3_layers[:5]))
        self.resnet3_bottleneck3_relu = resnet3_bottleneck3_layers[6] 
        self.attention3_3 = Attention(1024, 1024//16)

        #   Bottleneck: 7-4
        resnet3_bottleneck4_layers = list(resnet3_layers[3].children())
        self.resnet3_bottleneck4_pre = nn.Sequential(*(resnet3_bottleneck4_layers[:5]))
        self.resnet3_bottleneck4_relu = resnet3_bottleneck4_layers[6] 
        self.attention3_4 = Attention(1024, 1024//16)

        #   Bottleneck: 7-5
        resnet3_bottleneck5_layers = list(resnet3_layers[4].children())
        self.resnet3_bottleneck5_pre = nn.Sequential(*(resnet3_bottleneck5_layers[:5]))
        self.resnet3_bottleneck5_relu = resnet3_bottleneck5_layers[6] 
        self.attention3_5 = Attention(1024, 1024//16)

        #   Bottleneck: 7-6
        resnet3_bottleneck6_layers = list(resnet3_layers[5].children())
        self.resnet3_bottleneck6_pre = nn.Sequential(*(resnet3_bottleneck6_layers[:5]))
        self.resnet3_bottleneck6_relu = resnet3_bottleneck6_layers[6] 
        self.attention3_6 = Attention(1024, 1024//16)

        # Resnet4: 8
        resnet4_layers = list(layers[7].children())

        #   Bottleneck: 8-1
        resnet4_bottleneck1_layers = list(resnet4_layers[0].children())
        self.resnet4_bottleneck1_pre = nn.Sequential(*(resnet4_bottleneck1_layers[:5])) # Conv2d > BN > Conv2d > BN > Conv2d > BN
        self.resnet4_bottleneck1_skip = nn.Sequential(*(resnet4_bottleneck1_layers[7])) # Sequntial(Conv2d > BN)
        self.resnet4_bottleneck1_relu = resnet4_bottleneck1_layers[6]
        self.attention4_1 = Attention(2048, 2048//16)

        #   Bottleneck: 7-2
        resnet4_bottleneck2_layers = list(resnet4_layers[1].children())
        self.resnet4_bottleneck2_pre = nn.Sequential(*(resnet4_bottleneck2_layers[:5]))
        self.resnet4_bottleneck2_relu = resnet4_bottleneck2_layers[6] 
        self.attention4_2 = Attention(2048, 2048//16)

        #   Bottleneck: 7-3
        resnet4_bottleneck3_layers = list(resnet4_layers[2].children())
        self.resnet4_bottleneck3_pre = nn.Sequential(*(resnet4_bottleneck3_layers[:5]))
        self.resnet4_bottleneck3_relu = resnet4_bottleneck3_layers[6] 
        self.attention4_3 = Attention(2048, 2048//16)

        # post layers
        self.post_layers = nn.Sequential(*(layers[8:9]))
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
        if self.use_attention:
            y = self.attention1_1(y)
        x = self.resnet1_bottleneck1_relu(y + z)

        #   Bottleneck2
        y = self.resnet1_bottleneck2_pre(x)
        if self.use_attention:
            y = self.attention1_2(y)
        x = self.resnet1_bottleneck2_relu(x + y)

        #   Bottleneck3
        y = self.resnet1_bottleneck3_pre(x)
        if self.use_attention:
            y = self.attention1_3(y)
        x = self.resnet1_bottleneck3_relu(x + y)

        # Resnet2
        #   Bottleneck1
        y = self.resnet2_bottleneck1_pre(x)
        z = self.resnet2_bottleneck1_skip(x)
        if self.use_attention:
            y = self.attention2_1(y)
        x = self.resnet2_bottleneck1_relu(y + z)

        #   Bottleneck2
        y = self.resnet2_bottleneck2_pre(x)
        if self.use_attention:
            y = self.attention2_2(y)
        x = self.resnet2_bottleneck2_relu(x + y)

        #   Bottleneck3
        y = self.resnet2_bottleneck3_pre(x)
        if self.use_attention:
            y = self.attention2_3(y)
        x = self.resnet2_bottleneck3_relu(x + y)

        #   Bottleneck4
        y = self.resnet2_bottleneck4_pre(x)
        if self.use_attention:
            y = self.attention2_4(y)
        x = self.resnet2_bottleneck4_relu(x + y)

        # Resnet3
        #   Bottleneck1
        y = self.resnet3_bottleneck1_pre(x)
        z = self.resnet3_bottleneck1_skip(x)
        if self.use_attention:
            y = self.attention3_1(y)
        x = self.resnet3_bottleneck1_relu(y + z)

        #   Bottleneck2
        y = self.resnet3_bottleneck2_pre(x)
        if self.use_attention:
            y = self.attention3_2(y)
        x = self.resnet3_bottleneck2_relu(x + y)

        #   Bottleneck3
        y = self.resnet3_bottleneck3_pre(x)
        if self.use_attention:
            y = self.attention3_3(y)
        x = self.resnet3_bottleneck3_relu(x + y)

        #   Bottleneck4
        y = self.resnet3_bottleneck4_pre(x)
        if self.use_attention:
            y = self.attention3_4(y)
        x = self.resnet3_bottleneck4_relu(x + y)

        #   Bottleneck5
        y = self.resnet3_bottleneck5_pre(x)
        if self.use_attention:
            y = self.attention3_5(y)
        x = self.resnet3_bottleneck5_relu(x + y)

        #   Bottleneck6
        y = self.resnet3_bottleneck6_pre(x)
        if self.use_attention:
            y = self.attention3_6(y)
        x = self.resnet3_bottleneck6_relu(x + y)

        # Resnet4
        #   Bottleneck1
        y = self.resnet4_bottleneck1_pre(x)
        z = self.resnet4_bottleneck1_skip(x)
        if self.use_attention:
            y = self.attention4_1(y)
        x = self.resnet4_bottleneck1_relu(y + z)

        #   Bottleneck2
        y = self.resnet4_bottleneck2_pre(x)
        if self.use_attention:
            y = self.attention4_2(y)
        x = self.resnet4_bottleneck2_relu(x + y)

        #   Bottleneck3
        y = self.resnet4_bottleneck3_pre(x)
        if self.use_attention:
            y = self.attention4_3(y)
        x = self.resnet4_bottleneck3_relu(x + y)

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
