from torch import nn
import torch.nn.init as init
from utils.group_normalization import GroupNorm2d

# Weight_standardization
class conv3x3_WS(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True) :
        super(conv3x3_WS, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def foward(self, x):
        weight = self.weight
        weight_mean = torch.mean(weight, dim = (1,2,3), keepdim = True)
        weight_var = torch.var(weight, dim = (1,2,3), keepdim = True)

        weight = (weight - mean) / torch.sqrt(var + 1e-5 )

        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class SELayer(nn.Module):
    def __init__(self, channel, reduction = 16):
        super(SELayer,self).__init__()
        # AdaptiveAvgPool target size  = 1 x 1 x C 
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(channel // reduction, channel, bias = False),
            nn.Sigmoid()
        )
    def forward(self, x):

        out = self.avg_pool(x)
        out = out.view(out.size(0), out.size(1))
        out = self.fc(out)
        out = out.view(out.size(0), out.size(1), 1,1)
        out = out.expand_as(x)
        return x * out

class CifarSEResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, reduction = 16):
        super(CifarSEResidualBlock, self).__init__()
        self.conv1 = conv3x3_WS(in_channels, out_channels, stride = stride)
        self.gn1 = nn.GroupNorm(num_groups = 8, num_channels = out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = conv3x3_WS(out_channels, out_channels)
        self.gn2 = nn.GroupNorm(num_groups = 8, num_channels = out_channels)

        self.se = SELayer(out_channels, reduction)

        if in_channels != out_channels or stride != 1 :
            self.down_sample = nn.Sequential(
                                                conv3x3_WS(in_channels, out_channels, kernel_size = 3, padding = 1,  stride = stride),
                                                # nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1, stride = stride, bias = False),
                                                nn.GroupNorm(num_groups = 8, num_channels = out_channels))
        else :  self.down_sample = None

        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.gn2(out)
        out = self.se(out)

        if self.down_sample is not None :
            x = self.down_sample(x)
    
        out = out + x
        out = self.relu(out)

        return out

class CifarSEResNet(nn.Module):        
    def __init__ (self, n_layers, block, num_classes = 10, reduction = 16):
            super(CifarSEResNet, self).__init__()
            self.conv1 = conv3x3_WS(in_channels = 3, out_channels = 16, stride = 1)
            self.gn1 = nn.GroupNorm(num_groups = 8, num_channels = 16)
            self.relu = nn.ReLU(inplace = True)
                
            self.layers1 = self._make_layers(block, 16, 16, stride = 1, reduction = reduction)
            self.layers2 = self._make_layers(block, 16, 32, stride = 2, reduction = reduction)
            self.layers3 = self._make_layers(block, 32, 64, stride = 2, reduction = reduction)

            self.avg_pooling = nn.AvgPool2d(8, stride = 1)
            self.fc_out = nn.Linear(64, num_classes)

            for m in self.modules(): 
                if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                    init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    init.constant_(m.weight, 1)
                    init.constant_(m.bias, 0)

        # n = # of layers
    def _make_layers(self, block, in_channels, out_channels, stride, reduction, n = 5):

        layers = nn.ModuleList([block(in_channels, out_channels, stride = stride, reduction = reduction)])

        for _ in range(n - 1):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        out = self.layers1(out)
        out = self.layers2(out)
        out = self.layers3(out)

        out = self.avg_pooling(out)
        out = out.view(out.size(0), -1)
        out = self.fc_out(out)

        return out

def SEresnet_gn_ws():
    return CifarSEResNet(5, block = CifarSEResidualBlock, reduction = 16)