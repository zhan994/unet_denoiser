"""
  U-Net Model
  Zhihao Zhan(zhanzhihao_dt@163.com)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class double_conv(nn.Module):
    def __init__(self, input_channels, output_channels) -> None:
        super(double_conv, self).__init__()
        self.conv1 = nn.Conv2d(
            input_channels, output_channels, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(
            output_channels, output_channels, kernel_size=3, padding='same')
        self.relu_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(self.relu_activation(x))
        x = self.conv2(self.relu_activation(x))
        return x


class traspose_conv(nn.Module):
    def __init__(self, num_of_channels):
        super(traspose_conv, self).__init__()
        self.trasnpose_conv = nn.ConvTranspose2d(
            num_of_channels, int(num_of_channels / 2), kernel_size=2, stride=2)

    def forward(self, x):
        x = self.trasnpose_conv(x)
        return x


class double_decoder_conv(nn.Module):
    def __init__(self, input_channels1, output_channels1, output_channels2):
        super(double_decoder_conv, self).__init__()
        self.conv1 = nn.Conv2d(
            input_channels1, output_channels1, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(
            output_channels1, output_channels2, kernel_size=3, padding='same')
        self.relu_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(self.relu_activation(x))
        x = self.conv2(self.relu_activation(x))
        return x


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.double_conv1 = double_conv(1, 64)
        self.double_conv2 = double_conv(64, 128)
        self.double_conv3 = double_conv(128, 256)
        self.double_conv4 = double_conv(256, 512)
        self.double_conv5 = double_conv(512, 1024)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.traspose_conv1 = traspose_conv(1024)
        self.traspose_conv2 = traspose_conv(512)
        self.traspose_conv3 = traspose_conv(256)
        self.traspose_conv4 = traspose_conv(128)

        self.double_decoder_conv1 = double_decoder_conv(1024, 512, 512)
        self.double_decoder_conv2 = double_decoder_conv(512, 256, 256)
        self.double_decoder_conv3 = double_decoder_conv(256, 128, 128)
        self.double_decoder_conv4 = double_decoder_conv(128, 64, 64)

        self.final_conv = nn.Conv2d(64, 1, kernel_size=1, padding='same')

    def forward(self, x):
        conv_output1 = self.double_conv1(x)
        conv_output2 = self.double_conv2(self.maxpool(conv_output1))
        conv_output3 = self.double_conv3(self.maxpool(conv_output2))
        conv_output4 = self.double_conv4(self.maxpool(conv_output3))
        x = self.double_conv5(self.maxpool(conv_output4))

        x = self.traspose_conv1(x)
        x = torch.cat([x, conv_output4], dim=1)
        x = self.double_decoder_conv1(x)

        x = self.traspose_conv2(x)
        x = torch.cat([x, conv_output3], dim=1)
        x = self.double_decoder_conv2(x)

        x = self.traspose_conv3(x)
        x = torch.cat([x, conv_output2], dim=1)
        x = self.double_decoder_conv3(x)

        x = self.traspose_conv4(x)
        x = torch.cat([x, conv_output1], dim=1)
        x = self.double_decoder_conv4(x)

        x = self.final_conv(x)

        return x
