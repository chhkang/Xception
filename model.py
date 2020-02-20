import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class pre_flow(nn.Module):
    # 맨위의 conv 32가 변형된 모델
    def __init__(self):
        super(pre_flow, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = SeparableConv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = SeparableConv2d(128, 128, kernel_size=3, padding=1)
        self.shortcut = nn.Conv2d(64, 128, kernel_size=1, stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        s_out = self.shortcut(out)
        out = self.bn3(self.conv3(out))
        out = self.bn3(self.conv4(F.relu(out)))
        out = self.maxpool(out)
        return s_out + out


class entry_flow(nn.Module):
    def __init__(self, input, output):
        super(entry_flow, self).__init__()
        self.conv1 = SeparableConv2d(input, output, kernel_size=3, padding=1)
        self.conv2 = SeparableConv2d(output, output, kernel_size=3, padding=1)
        self.shortcut = nn.Conv2d(input, output, kernel_size=1, stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.bn = nn.BatchNorm2d(output)

    def forward(self, x):
        out = self.bn(self.conv1(F.relu(x)))
        out = self.bn(self.conv2(F.relu(out)))
        out = self.maxpool(out)
        x = self.shortcut(x)
        return out + x


class middle_flow(nn.Module):
    def __init__(self):
        super(middle_flow, self).__init__()
        self.conv = SeparableConv2d(728, 728, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(728)

    def forward(self, x):
        out = self.bn(self.conv(F.relu(x)))
        out = self.bn(self.conv(F.relu(out)))
        out = self.bn(self.conv(F.relu(out)))
        return out + x


class exit_flow(nn.Module):
    def __init__(self):
        super(exit_flow, self).__init__()
        self.conv1 = SeparableConv2d(728, 728, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(728)
        self.conv2 = SeparableConv2d(728, 1024, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(1024)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.shortcut = nn.Conv2d(728, 1024, kernel_size=1, stride=2)
        self.conv3 = SeparableConv2d(1024, 2048, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(2048)
        self.avgpool = nn.AvgPool2d(kernel_size=2)
        self.linear = nn.Linear(2048, 100)

    def forward(self, x):
        out = self.bn1(self.conv1(F.relu(x)))
        out = self.bn2(self.conv2(F.relu(out)))
        out = self.maxpool(out)
        x = self.shortcut(x)
        out = out + x
        out = self.bn3(self.conv3(out))
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class XceptionNet(nn.Module):
    def __init__(self):
        super(XceptionNet, self).__init__()
        self.model = self._make_layer()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def _make_layer(self):
        layers = []
        layers.append(pre_flow())
        layers.append(entry_flow(128, 256))
        layers.append(entry_flow(256, 728))

        for i in range(8):
            layers.append(middle_flow())

        layers.append(exit_flow())
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
