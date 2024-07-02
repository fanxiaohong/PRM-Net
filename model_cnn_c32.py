import torch.nn as nn
import torch.nn.functional as F
import torch

###################################################################################
class InceptionModule(nn.Module):
    def __init__(self, in_channels, f1_channels, f2_channels, f3_channels, f4_channels):
        super(InceptionModule, self).__init__()

        # 1x1 Conv branch
        self.branch1 = nn.Conv3d(in_channels, f1_channels, kernel_size=3, padding=1, dilation = 1, bias=False)

        # 3x3 Conv branch
        self.branch2 = nn.Conv3d(in_channels, f2_channels, kernel_size=3, padding=2, dilation = 2, bias=False)

        # 5x5 Conv branch
        self.branch3 = nn.Conv3d(in_channels, f3_channels, kernel_size=3, padding=3, dilation = 3, bias=False)

        # Max pooling branch
        self.branch4 = nn.Conv3d(in_channels, f4_channels, kernel_size=3, padding=4, dilation = 4, bias=False)

    def forward(self, x):
        out1 = self.branch1(x)
        # print('out1 shape', out1.shape)
        out2 = self.branch2(x)
        # print('out2 shape', out2.shape)
        out3 = self.branch3(x)
        # print('out3 shape', out3.shape)
        out4 = self.branch4(x)
        # print('out4 shape', out4.shape)
        return torch.cat([out1, out2, out3, out4], dim=1)
###################################################################################
class feature_extrator(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(feature_extrator, self).__init__()
        self.incept = InceptionModule(in_ch, out_ch, out_ch, out_ch, out_ch)
        self.bn1 = nn.BatchNorm3d(out_ch * 4)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.ac = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(out_ch*4, out_ch, kernel_size=3, padding=1, dilation=1, bias=False)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, dilation=1, bias=False)
        self.conv3 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, dilation=1, bias=False)

    def forward(self, x):
        x = self.incept(x)
        x = self.bn1(x)
        x = self.ac(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x1 = self.ac(x)

        # inner residual
        x = self.conv2(x1)
        x = self.bn2(x)
        x = self.ac(x)
        x = self.conv3(x)
        x = self.bn2(x) + x1

        return x
###################################################################################
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.feat = 32
        self.ex1 = feature_extrator(3, self.feat)
        self.ex2 = feature_extrator(self.feat, self.feat)
        self.ex3 = feature_extrator(self.feat, self.feat)
        self.ex4 = feature_extrator(self.feat, self.feat)
        self.ex5 = feature_extrator(self.feat, self.feat)
        self.down = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.fc_final = nn.Sequential(nn.Linear(self.feat, 50),
                                      nn.Linear(50, 50),
                                      nn.Linear(50, 2)
                                      )

    def forward(self, x_ori):
        x = x_ori
        x = self.ex1(x)
        x = self.down(x)
        # print('x shape', x.shape)
        x = self.ex2(x)+x
        x = self.down(x)
        # print('x shape', x.shape)
        x = self.ex3(x)+x
        x = self.down(x)
        # print('x shape', x.shape)
        x = self.ex4(x)+x
        x = self.down(x)
        # print('x shape', x.shape)
        x = self.ex5(x)+x
        # print('x shape', x.shape)
        x = self.down(x)
        x_feature = x.view(x.size(0), -1)
        x = self.fc_final(x_feature)
        x = nn.Sigmoid()(x)
        return x,x_feature





