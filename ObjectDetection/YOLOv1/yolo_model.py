import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.batchnorm = nn.BatchNorm2d(out_channels) #Original paper did not have batch norm
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        out = self.conv(x)
        out = self.batchnorm(out)
        out = self.leakyrelu(out)
        return out

class YOLOv1(nn.Module):
    def __init__(self, in_channels=3, split_size, num_boxes, num_classes):
        super(YOLOv1, self).__init__()
        self.in_channels = in_channels

        self.detect = nn.Sequential(
            ConvBlock(self.in_channels, 64, 7, 2, 3)
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            ConvBlock(64, 192, 3, 1, 1)
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            ConvBlock(in_channels=192, out_channels=128, kernel_size=1, stride=1, padding=0)
            ConvBlock(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
            ConvBlock(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0)
            ConvBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            ConvBlock(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0) #first
            ConvBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
            ConvBlock(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0) #second
            ConvBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
            ConvBlock(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0) #third
            ConvBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
            ConvBlock(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0) #fourth
            ConvBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
            ConvBlock(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0)
            ConvBlock(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            ConvBlock(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0) #first
            ConvBlock(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
            ConvBlock(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0) #second
            ConvBlock(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
            ConvBlock(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1)
            ConvBlock(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1)
            ConvBlock(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1)
            ConvBlock(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1)
        )

        S, B, C = split_size, num_boxes, num_classes
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 512),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(512, S * S * (C + B * 5)),
        )

    def forward(self, x):
        x = self.detect(x)
        return self.fcs()#torch.flatten(x, start_dim=1))