import torch.nn as nn
import math
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torch


__all__ = ['ResNetSSD', 'resnetssd18', 'resnetssd34', 'resnetssd50', 'resnetssd101', 'resnetssd152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def hw_flattern(x):
    return x.view(x.size()[0],x.size()[1],-1)


class ResNetSSD(nn.Module):

    def __init__(self, block, layers, num_classes=2):
        self.inplanes = 64
        super(ResNetSSD, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        self.new_layer1 = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                                        nn.BatchNorm2d(512),
                                        nn.ReLU(inplace=True))

        self.new_layer2 = nn.Sequential(nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(inplace=True))

        self.num_classes = num_classes
        # Lateral layers
        self.latlayer1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)  
        self.latlayer2 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0)
        # Top-down layers
        self.toplayer1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        
        
        self.conf_p1 = nn.Conv2d(256,  4 * num_classes, kernel_size=3, padding=1)
        self.conf_p2 = nn.Conv2d(256,  6 * num_classes, kernel_size=3, padding=1)
        self.conf_p3 = nn.Conv2d(256,  6 * num_classes, kernel_size=3, padding=1)
        self.conf_p4 = nn.Conv2d(256,  6 * num_classes, kernel_size=3, padding=1)
        self.conf_p5 = nn.Conv2d(256,  6 * num_classes, kernel_size=3, padding=1)
        self.conf_c6 = nn.Conv2d(256,  6 * num_classes, kernel_size=3, padding=1)

        self.locs_p1 = nn.Conv2d(256,  4 * 4, kernel_size=3, padding=1)
        self.locs_p2 = nn.Conv2d(256,  6 * 4, kernel_size=3, padding=1)
        self.locs_p3 = nn.Conv2d(256,  6 * 4, kernel_size=3, padding=1)
        self.locs_p4 = nn.Conv2d(256,  6 * 4, kernel_size=3, padding=1)
        self.locs_p5 = nn.Conv2d(256,  6 * 4, kernel_size=3, padding=1)
        self.locs_c6 = nn.Conv2d(256,  6 * 4, kernel_size=3, padding=1)



    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
        
    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y
        
    def locs_forward(self, p1, p2, p3, p4, p5, c6):
        p1_locs = self.locs_p1(p1).permute(0, 2, 3, 1).contiguous().view(p1.shape[0], -1, 4)
        p2_locs = self.locs_p2(p2).permute(0, 2, 3, 1).contiguous().view(p2.shape[0], -1, 4)
        p3_locs = self.locs_p3(p3).permute(0, 2, 3, 1).contiguous().view(p3.shape[0], -1, 4)
        p4_locs = self.locs_p4(p4).permute(0, 2, 3, 1).contiguous().view(p4.shape[0], -1, 4)
        p5_locs = self.locs_p5(p5).permute(0, 2, 3, 1).contiguous().view(p5.shape[0], -1, 4)
        c6_locs = self.locs_c6(c6).permute(0, 2, 3, 1).contiguous().view(c6.shape[0], -1, 4)
        return torch.cat([p1_locs, p2_locs, p3_locs, p4_locs, p5_locs, c6_locs], dim=1)

    def conf_forward(self, p1, p2, p3, p4, p5, c6):
        p1_conf = self.conf_p1(p1).permute(0, 2, 3, 1).contiguous().view(p1.shape[0], -1, self.num_classes)
        p2_conf = self.conf_p2(p2).permute(0, 2, 3, 1).contiguous().view(p2.shape[0], -1, self.num_classes)
        p3_conf = self.conf_p3(p3).permute(0, 2, 3, 1).contiguous().view(p3.shape[0], -1, self.num_classes)
        p4_conf = self.conf_p4(p4).permute(0, 2, 3, 1).contiguous().view(p4.shape[0], -1, self.num_classes)
        p5_conf = self.conf_p5(p5).permute(0, 2, 3, 1).contiguous().view(p5.shape[0], -1, self.num_classes)
        c6_conf = self.conf_c6(c6).permute(0, 2, 3, 1).contiguous().view(c6.shape[0], -1, self.num_classes)
        return torch.cat([p1_conf, p2_conf, p3_conf, p4_conf, p5_conf, c6_conf], dim=1)

    def forward(self, x):
        c0 = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        c1 = x
        x = self.maxpool(x)       
        x = self.layer1(x)
        c2 = x
        x = self.layer2(x)
        c3 = x
        x = self.layer3(x)
        c4 = x
        x = self.new_layer1(x)
        c5 = x        
        x = self.new_layer2(x)
        c6 = x                 #256   
        
        # build top-down path
        p5 = self._upsample_add(c6, self.latlayer1(c5))  #256 
        p5 = self.toplayer1(p5)
        #print('p5',p5.shape)
        p4 = self._upsample_add(p5, self.latlayer2(c4))
        p4 = self.toplayer1(p4)
        #print(p4.shape)
        p3 = self._upsample_add(p4, self.latlayer1(c3))
        p3 = self.toplayer1(p3)
        #print(p3.shape)
        p2 = self._upsample_add(p3, c2)
        p2 = self.toplayer1(p2)
        #print(p2.shape)
        p1 = self._upsample_add(p2, self.latlayer3(c1))
        p1 = self.toplayer1(p1)
        #print(p1.shape)
        
        locs = self.locs_forward(p1, p2, p3, p4, p5, c6)
        conf = self.conf_forward(p1, p2, p3, p4, p5, c6)

        return (locs, conf, [c0, c1, c2, c3, c4])


def resnetssd18(pretrained=False, num_classes=2):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetSSD(BasicBlock, [2, 2, 2, 2], num_classes)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir='.'), strict=False)
    return model


def resnetssd34(pretrained=False, num_classes=2):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetSSD(BasicBlock, [3, 4, 6, 3], num_classes)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir='.'), strict=False)
    return model


def resnetssd50(pretrained=False, num_classes=2):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetSSD(Bottleneck, [3, 4, 6, 3], num_classes)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='/data2/coldplay/ANCIS'), strict=False)
    return model


def resnetssd101(pretrained=False, num_classes=2):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetSSD(Bottleneck, [3, 4, 23, 3], num_classes)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir='.'), strict=False)
    return model


def resnetssd152(pretrained=False, num_classes=2):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetSSD(Bottleneck, [3, 8, 36, 3], num_classes)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152'], model_dir='.'), strict=False)
    return model
