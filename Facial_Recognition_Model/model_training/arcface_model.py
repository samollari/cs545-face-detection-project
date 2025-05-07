import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as md

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.PReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class IRSEBlock(nn.Module):
    expansion = 1  # Not 4 like standard ResNet, ArcFace uses 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, use_se=True):
        super(IRSEBlock, self).__init__()
        self.bn0 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(out_channels)

    def forward(self, x):
        identity = x
        out = self.bn0(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_se:
            out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.prelu(out)
        return out

class ResNetIRSE(nn.Module):
    def __init__(self, layers, use_se=True, dropout=0.4, embedding_size=512):
        super(ResNetIRSE, self).__init__()
        self.in_channels = 64
        self.use_se = use_se

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.prelu = nn.PReLU(self.in_channels)

        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        self.bn2 = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout(p=dropout)
        self.flatten = nn.Flatten()
        # self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc = nn.Linear(512 * 7 * 7, embedding_size)  # assuming input size 112x112
        self.bn3 = nn.BatchNorm1d(embedding_size)

        self._initialize_weights()

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        layers = []
        layers.append(IRSEBlock(self.in_channels, out_channels, stride, downsample, use_se=self.use_se))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(IRSEBlock(self.in_channels, out_channels, use_se=self.use_se))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        #x = self.pool(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.bn3(x)
        return x

def resnet50_IR_SE(embedding_size=512):
    return ResNetIRSE([3, 4, 14, 3], use_se=True, embedding_size=embedding_size)

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.5, easy_margin=False):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m
        self.easy_margin = easy_margin

        self.register_buffer('cos_m', torch.tensor(math.cos(m)))
        self.register_buffer('sin_m', torch.tensor(math.sin(m)))
        self.register_buffer('th', torch.tensor(math.cos(math.pi - m)))
        self.register_buffer('mm', torch.tensor(math.sin(math.pi - m) * m))

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        cosine = cosine.clamp(-1.0 + 1e-5, 1.0 - 1e-5)
        
        # Only apply margin to the correct class
        phi = cosine * self.cos_m - torch.sqrt(1.0 - cosine**2 + 1e-7) * self.sin_m


        # Only modify the target class
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        output = one_hot * phi + (1.0 - one_hot) * cosine
        return output * self.s


class ArcFaceModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = resnet50_IR_SE()
        self.backbone.fc = nn.Identity()
        self.backbone.bn3 = nn.Identity()
        
        self.embedding = nn.Sequential(
            nn.BatchNorm1d(100352),
            nn.Dropout(0.5),
            nn.Linear(100352, 512),
            nn.BatchNorm1d(512),
            nn.PReLU()
        )
        
        self.arc_margin = ArcMarginProduct(512,num_classes)

    def forward(self, x, labels=None):        
        x = self.backbone(x)
        x = self.embedding(x)
        x = F.normalize(x, dim=1)
        if labels is not None:
            return self.arc_margin(x, labels)
        return x