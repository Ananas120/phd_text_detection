"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
#from torchvision.models.mobilenet import mobilenet_v2, InvertedResidual

class Base(nn.Module):
    def __init__(self):
        super().__init__()

    def init_weights(self):
        layers = [*self.additional_blocks, *self.loc, *self.conf]
        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

    def bbox_view(self, src, loc, conf):
        ret = []
        for s, l, c in zip(src, loc, conf):
            ret.append((l(s).view(s.size(0), 4, -1), c(s).view(s.size(0), self.num_classes, -1)))

        locs, confs = list(zip(*ret))
        locs, confs = torch.cat(locs, 2).contiguous(), torch.cat(confs, 2).contiguous()
        return locs, confs


class ResNet(nn.Module):
    def __init__(self, backbone="resnet50"):
        super().__init__()
        if backbone == 'resnet18':
            backbone = resnet18(pretrained=True)
            self.out_channels = [256, 512, 512, 256, 256, 128]
        elif backbone == 'resnet34':
            backbone = resnet34(pretrained=True)
            self.out_channels = [256, 512, 512, 256, 256, 256]
        elif backbone == 'resnet50':
            backbone = resnet50(pretrained=True)
            self.out_channels = [1024, 512, 512, 256, 256, 256]
        elif backbone == 'resnet101':
            backbone = resnet101(pretrained=True)
            self.out_channels = [1024, 512, 512, 256, 256, 256]
        else:  # backbone == 'resnet152':
            backbone = resnet152(pretrained=True)
            self.out_channels = [1024, 512, 512, 256, 256, 256]
        
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:7])

        conv4_block1 = self.feature_extractor[-1][0]
        conv4_block1.conv1.stride = (1, 1)
        conv4_block1.conv2.stride = (1, 1)
        conv4_block1.downsample[0].stride = (1, 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        return x


class SSD(Base):
    def __init__(self,model,trunc, backbone=ResNet(), figsize=300, num_classes=81):
        super().__init__()

        self.feature_extractor = backbone
        self.num_classes = num_classes
        self.figsize = figsize
        self.trunc = trunc
        
        if "custom" in model:
            self.num_defaults = [3, 4, 4, 4, 3, 3]
        else:
            self.num_defaults = [4, 6, 6, 6, 4, 4] #original SSD


        if self.figsize == 512:
            self.feature_extractor.out_channels.append(256)
            if "custom" in model:
                self.num_defaults.append(3)
            else:
                self.num_defaults.append(4)

        if self.trunc:
            self.num_defaults = self.num_defaults[:3] #keep only 3 first layers
        
        self._build_additional_features(self.feature_extractor.out_channels)
        
        self.loc = []
        self.conf = []

        for nd, oc in zip(self.num_defaults, self.feature_extractor.out_channels):
            self.loc.append(nn.Conv2d(oc, nd * 4, kernel_size=3, padding=1))
            self.conf.append(nn.Conv2d(oc, nd * self.num_classes, kernel_size=3, padding=1))

        self.loc = nn.ModuleList(self.loc)
        self.conf = nn.ModuleList(self.conf)
        self.init_weights()

    def _build_additional_features(self, input_size):
        self.additional_blocks = []
    
        if self.figsize == 300:    
            for i, (input_size, output_size, channels) in enumerate(zip(input_size[:-1], input_size[1:], [256, 256, 128, 128, 128])):
                if i < 3:
                    layer = nn.Sequential(
                        nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
                        nn.BatchNorm2d(channels),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(channels, output_size, kernel_size=3, padding=1, stride=2, bias=False),
                        nn.BatchNorm2d(output_size),
                        nn.ReLU(inplace=True),
                    )
                else:
                    layer = nn.Sequential(
                        nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
                        nn.BatchNorm2d(channels),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(channels, output_size, kernel_size=3, bias=False),
                        nn.BatchNorm2d(output_size),
                        nn.ReLU(inplace=True),
                    )

                self.additional_blocks.append(layer)
        else:
            for i, (input_size, output_size, channels) in enumerate(zip(input_size[:-1], input_size[1:], [256, 256, 128, 128, 128,128])):
                k2 = 3
                if i == 5:
                    k2 = 4
                layer = nn.Sequential(
                    nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, output_size, kernel_size=k2, padding=1, stride=2, bias=False),
                    nn.BatchNorm2d(output_size),
                    nn.ReLU(inplace=True),
                )

                self.additional_blocks.append(layer)
                
        if self.trunc:
            self.additional_blocks = self.additional_blocks[:2]

        self.additional_blocks = nn.ModuleList(self.additional_blocks)
        

    def forward(self, x):
        x = self.feature_extractor(x)
        detection_feed = [x]
        for l in self.additional_blocks:
            x = l(x)
            detection_feed.append(x)
        locs, confs = self.bbox_view(detection_feed, self.loc, self.conf)
        return locs, confs
    

class Textboxes(Base):
    def __init__(self, model, trunc, backbone=ResNet(),figsize = 300, num_classes=2):
        super().__init__()

        self.feature_extractor = backbone
        self.num_classes = num_classes
        self.figsize = figsize
        self.trunc = trunc

        if model == "TB":
            n_default = 12
        else:
            n_default = 7 #no offset + extra scale

            
        self.num_defaults = [n_default]*6
        
        if self.figsize == 512:
            self.feature_extractor.out_channels.append(256)
            self.num_defaults.append(n_default)

        if self.trunc:
            self.num_defaults = self.num_defaults[:3] #keep only 3 first layers

        self._build_additional_features(self.feature_extractor.out_channels)
        
        self.loc = []
        self.conf = []

        for nd, oc in zip(self.num_defaults, self.feature_extractor.out_channels):
            self.loc.append(nn.Conv2d(oc, nd * 4, kernel_size=(1,5), padding=(0,2)))
            self.conf.append(nn.Conv2d(oc, nd * self.num_classes, kernel_size=(1,5), padding=(0,2)))

        self.loc = nn.ModuleList(self.loc)
        self.conf = nn.ModuleList(self.conf)
        self.init_weights()

    def get_n_params(self):
        pp=0
        for p in list(self.parameters()):
            nn=1
            for s in list(p.size()):
                nn = nn*s
            pp += nn
        return pp

    def _build_additional_features(self, input_size):
        self.additional_blocks = []
    
        if self.figsize == 300:    
            for i, (input_size, output_size, channels) in enumerate(zip(input_size[:-1], input_size[1:], [256, 256, 128, 128, 128])):
                if i < 3:
                    layer = nn.Sequential(
                        nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
                        nn.BatchNorm2d(channels),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(channels, output_size, kernel_size=3, padding=1, stride=2, bias=False),
                        nn.BatchNorm2d(output_size),
                        nn.ReLU(inplace=True),
                    )
                else:
                    layer = nn.Sequential(
                        nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
                        nn.BatchNorm2d(channels),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(channels, output_size, kernel_size=3, bias=False),
                        nn.BatchNorm2d(output_size),
                        nn.ReLU(inplace=True),
                    )

                self.additional_blocks.append(layer)
        else:
            for i, (input_size, output_size, channels) in enumerate(zip(input_size[:-1], input_size[1:], [256, 256, 128, 128, 128,128])):
                k2 = 3
                if i == 5:
                    k2 = 4
                layer = nn.Sequential(
                    nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, output_size, kernel_size=k2, padding=1, stride=2, bias=False),
                    nn.BatchNorm2d(output_size),
                    nn.ReLU(inplace=True),
                )

                self.additional_blocks.append(layer)

        if self.trunc:
            self.additional_blocks = self.additional_blocks[:2]

        self.additional_blocks = nn.ModuleList(self.additional_blocks)


    def forward(self, x):
        x = self.feature_extractor(x)
        detection_feed = [x]
        for l in self.additional_blocks:
            x = l(x)
            detection_feed.append(x)
        locs, confs = self.bbox_view(detection_feed, self.loc, self.conf)
        return locs, confs

