import torch
import torch.nn as nn
import torch.nn.functional as F

#============================ MLP ===============================
class MLP(nn.Module):
    def __init__(self, config, weight_share=False, auxiliary=False, activate='relu', p=0):
        super(MLP, self).__init__()
        self.weight_share = weight_share
        self.auxiliary = auxiliary

        if activate == 'relu':
            self.activation = nn.ReLU()
        elif activate == 'tanh':
            self.activation = nn.Tanh()
        elif activate == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activate == 'selu':
            self.activation = nn.SELU()

        self.out_dims = config['out_dims']

        if self.auxiliary:
            assert self.weight_share==True
        
        if self.weight_share:
            layer1 = nn.Linear(1*14*14, self.out_dims[0])
        else:
            layer1 = nn.Linear(2*14*14, self.out_dims[0])
        layers = [layer1, self.activation, nn.Dropout(p=p)]

        for i in range(len(self.out_dims)-1):
            layers.append(nn.Linear(self.out_dims[i], self.out_dims[i+1]))
            layers.append(self.activation)
            layers.append(nn.Dropout(p=p))
        
        self.mlp = nn.Sequential(*layers)


        # The final output dim is 1. We are using BCELoss, Sigmoid() is required.
        # If output dim is larger than 1, please use CrossEntropyLoss, w/o Sigmoid().
        if weight_share:
            # Using weight sharing, siamese network
            # need to concate 2 features
            self.binary_cls = nn.Sequential(
                nn.Linear(self.out_dims[-1]*2, 1),
                nn.Sigmoid()
            )
        else:
            # Normal forward
            self.binary_cls = nn.Sequential(
                nn.Linear(self.out_dims[-1], 1),
                nn.Sigmoid()
            )

        self.digit_cls = nn.Sequential(
            nn.Linear(self.out_dims[-1], 10)
        )

    def forward(self, x):
        if self.weight_share:
            # Siamese Network, weight sharing
            x_1 = x[:, 0].unsqueeze(1)
            x_2 = x[:, 1].unsqueeze(1)

            x_1 = x_1.view(-1, 14*14)
            x_2 = x_2.view(-1, 14*14)

            f_1 = self.mlp(x_1)
            f_2 = self.mlp(x_2)

            f = torch.cat((f_1, f_2), 1)
            out_f = self.binary_cls(f)

            if self.auxiliary:
                out_1 = self.digit_cls(f_1)
                out_2 = self.digit_cls(f_2)
                return out_1, out_2, out_f
            else:
                # only weight sharing, but not auxiliary loss
                return out_f
        else:
            # Normal forward
            x = x.view(x.size(0), -1)
            x = self.mlp(x)
            x = self.binary_cls(x)
            return x


#============================ ConvNet ==========================
class ConvNet(nn.Module):
    def __init__(self, config, weight_share=False, auxiliary=False, activate='relu'):
        super(ConvNet, self).__init__()
        self.weight_share = weight_share
        self.auxiliary = auxiliary

        if activate == 'relu':
            self.activation = nn.ReLU()
        elif activate == 'tanh':
            self.activation = nn.Tanh()
        elif activate == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activate == 'selu':
            self.activation = nn.SELU()

        self.chns = config['chns']
        self.n_hid = config['n_hid']

        if self.weight_share:
            self.in_chn = 1
        else:
            self.in_chn = 2

        if self.auxiliary:
            assert self.weight_share==True

        self.feature = nn.Sequential(
            # CNN for feature extraction
            nn.Conv2d(self.in_chn, self.chns[0], kernel_size=3),                # N x 12 x 12
            #nn.BatchNorm2d(self.chns[0]),
            self.activation,
            nn.MaxPool2d(kernel_size=2, stride=2),                              # N x 6 x 6

            nn.Conv2d(self.chns[0], self.chns[1], kernel_size=3, padding=1),    # M x 6 x 6
            #nn.BatchNorm2d(self.chns[1]),
            self.activation,
            nn.MaxPool2d(kernel_size=2, stride=2),                              # M x 3 x 3

            nn.Conv2d(self.chns[1], self.chns[2], kernel_size=2, padding=1),    # K x 3 x 3
            #nn.BatchNorm2d(self.chns[2]),
            self.activation,
            nn.MaxPool2d(kernel_size=3, stride=3),                              # K x 1 x 1
        )

        # FC layers for digit classification
        self.digit_cls = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.chns[2], self.n_hid),
            self.activation,
            nn.Dropout(),
            nn.Linear(self.n_hid, 10),
        )

        # FC layers for binary classification
        if self.weight_share:
            self.binary_cls = nn.Sequential(
                nn.Dropout(),
                nn.Linear(self.chns[2] * 2, self.n_hid),
                self.activation,
                nn.Dropout(),
                nn.Linear(self.n_hid, 1),
                nn.Sigmoid()
            )
        else:
            self.binary_cls = nn.Sequential(
                nn.Dropout(),
                nn.Linear(self.chns[2], self.n_hid),
                self.activation,
                nn.Dropout(),
                nn.Linear(self.n_hid, 1),
                nn.Sigmoid()
            )

    def forward(self, x):
        if self.weight_share:
            # Siamese Network, weight sharing
            x_1 = x[:, 0].unsqueeze(1)
            x_2 = x[:, 1].unsqueeze(1)

            f_1 = self.feature(x_1)
            f_2 = self.feature(x_2)
            f_1 = f_1.view(f_1.size(0), -1)
            f_2 = f_2.view(f_2.size(0), -1)

            f = torch.cat((f_1, f_2), 1)
            out_f = self.binary_cls(f)

            if self.auxiliary:
                out_1 = self.digit_cls(f_1)
                out_2 = self.digit_cls(f_2)
                return out_1, out_2, out_f
            else:
                # only weight sharing, but not auxiliary loss
                return out_f
        else:
            # Normal forward
            x = self.feature(x)
            x = x.view(x.size(0), -1)
            x = self.binary_cls(x)
            return x


#================================== ResNet ==========================================
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, sk=True):
        super(BasicBlock, self).__init__()

        self.sk = sk

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = nn.Sequential(
                                nn.Conv2d(inplanes, planes, kernel_size=3),
                                nn.BatchNorm2d(planes)
                                )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.sk:
            identity = self.downsample(x)
            out += identity
        
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, sk=True):
        super(ResNet, self).__init__()
        self.sk = sk

        self.conv = nn.Conv2d(2, 64, kernel_size=3)     # Nx64x6x6
        self.maxpool = nn.MaxPool2d(kernel_size = 2)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.layer1 = BasicBlock(64, 64, self.sk)     # Nx64x6x6
        self.layer2 = BasicBlock(64, 128, self.sk)    # Nx128x2x2
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
            )

    def forward(self, x):
        x = self.maxpool(self.conv(x))
        x = self.bn(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        
        x = self.layer3(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    x = torch.rand((10, 2, 14, 14))
    model = ResNet(sk=False)
    print(model)
    out = model(x)