import torch.nn as nn
import torch.nn.functional as F

class pre_flow(nn.Module):
    def __init__(self):
        super(pre_flow,self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,128,kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.shortcut = nn.Conv2d(64, 128, kernel_size=1, stride=2)

    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        s_out = self.shortcut(out)
        out = F.relu(self.bn3(self.conv3(out)))
        out = self.bn3(self.conv3(F.relu(out)))
        return s_out + out

class entry_flow(nn.Module):
    def __init__(self, first, input, output):
        super(entry_flow, self).__init__()
        self.conv1 = nn.Conv2d(input, output, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(output, output, kernel_size=3, padding=1)
        self.shortcut = nn.Conv2d(input, output, kernel_size=1, stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2,ceil_mode=True)
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
        self.conv = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(512)

    def forward(self, x):
        out = self.bn(self.conv(F.relu(x)))
        out = self.bn(self.conv(F.relu(out)))
        out = self.bn(self.conv(F.relu(out)))
        return out + x


class exit_flow(nn.Module):
    def __init__(self):
        super(exit_flow, self).__init__()
        self.conv1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(1024)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.shortcut = nn.Conv2d(512, 1024, kernel_size=1, stride=2)

    def forward(self, x):
        out = self.bn1(self.conv1(F.relu(x)))
        out = self.bn2(self.conv2(F.relu(out)))
        self.maxpool(out)
        x = self.shortcut(x)
        return out + x


class XceptionNet(nn.Module):
    def __init__(self):
        super(XceptionNet, self).__init__()
        self.Entry = self._make_layer(1)
        self.Middle = self._make_layer(2)
        self.Exit = self._make_layer(3)

    def _make_layer(self,type):
        layers = []
        if type == 1:
            layers.append(pre_flow())
            layers.append(entry_flow(False, 128, 256))
            layers.append(entry_flow(False, 256, 512))
        elif type == 2:
            for i in range(8):
                layers.append(middle_flow())
        elif type == 3:
            layers.append(exit_flow())

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.Entry(out)
        out = self.Middle(out)
        out = self.Exit(out)
        out = out.view(out.size(0),-1)
        print(out.size())
        return out
