import torch
import torch.nn as nn
import torch.nn.functional as F


class entry_flow(nn.Module):
    def __init__(self, input, output):
        super(entry_flow, self).__init__()
        self.conv1 = nn.Conv2d(input, output, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(output, output, kernel_size=3, padding=1)
        self.shortcut = nn.Conv2d(input, output, kernel_size=1, stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        out = nn.BatchNorm2d(self.conv1(F.relu(x)))
        out = nn.BatchNorm2d(self.conv2(F.relu(out)))
        x = self.shortcut(x)
        return out + x


class middle_flow(nn.Module):
    def __init__(self):
        super(middle_flow, self).__init__()
        self.conv = nn.Conv2d(728, 728, kernel_size=3, padding=1)

    def forward(self, x):
        out = nn.BatchNorm2d(self.conv(F.relu(x)))
        out = nn.BatchNorm2d(self.conv(F.relu(out)))
        out = nn.BatchNorm2d(self.conv(F.relu(out)))
        return out + x


class exit_flow(nn.Module):
    def __init__(self):
        super(exit_flow, self).__init__()
        self.conv1 = nn.Conv2d(728, 728, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(728, 1024, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.shortcut = nn.Conv2d(728, 1024, kernel_size=1, stride=2)

    def forward(self, x):
        out = nn.BatchNorm2d(self.conv1(F.relu(x)))
        out = nn.BatchNorm2d(self.conv2(F.relu(out)))
        self.maxpool(out)
        x = self.shortcut(x)
        return out + x


class XceptionNet(nn.Module):
    def __init__(self):
        super(XceptionNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2)
        self.Entry = nn.Sequential(
            entry_flow(32, 128),
            entry_flow(128, 256),
            entry_flow(256, 728)
        )
        self.Middle = nn.Sequential(
            middle_flow(),
            middle_flow(),
            middle_flow(),
            middle_flow(),
            middle_flow(),
            middle_flow(),
            middle_flow(),
        )
        self.Exit = nn.Sequential(
            exit_flow(),
        )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.Entry(out)
        out = self.Middle(out)
        out = self.Exit(out)
        print(out.size())
        return out;
