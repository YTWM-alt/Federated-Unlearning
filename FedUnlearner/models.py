import torch.nn as nn
from torchvision.models import resnet18, resnet50

# [新增] 辅助函数：把模型里的 BN 层全换成 GN 层
def replace_bn_with_gn(module, num_groups=32):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            # GroupNorm 不需要 running_stats，非常适合 FL
            # num_channels 必须能被 num_groups 整除，ResNet 的通道通常是 64, 128, 256, 512，都能被 32 整除
            gn = nn.GroupNorm(num_groups, child.num_features)
            # 尝试保留一点 BN 的初始化参数（如果有的话）
            if child.affine:
                gn.weight.data.copy_(child.weight.data)
                gn.bias.data.copy_(child.bias.data)
            setattr(module, name, gn)
        else:
            replace_bn_with_gn(child, num_groups)

class SmallCNN(nn.Module):
    """一个很小的卷积网络，适合 MNIST (1×28×28) 和 CIFAR10 (3×32×32)，输入 224 也能跑"""
    def __init__(self, num_channels=1, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)  # 输出 (B,64,1,1)
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 展平为 (B,64)
        return self.classifier(x)


# Added required classes
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, output_padding=0,
                 activation_fn=nn.ReLU, batch_norm=True, transpose=False):
        if padding is None:
            padding = (kernel_size - 1) // 2
        model = []
        if not transpose:
            #             model += [ConvStandard(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding
            #                                 )]
            model += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                bias=not batch_norm)]
        else:
            model += [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                         output_padding=output_padding, bias=not batch_norm)]
        if batch_norm:
            model += [nn.BatchNorm2d(out_channels, affine=True)]
        model += [activation_fn()]
        super(Conv, self).__init__(*model)


class AllCNN(nn.Module):
    def __init__(self, num_channels=3, num_classes=10, filters_percentage=1, dropout=False, batch_norm=True):
        super().__init__()
        n_filter1 = int(96 * filters_percentage)
        n_filter2 = int(192 * filters_percentage)

        self.conv1 = Conv(num_channels, n_filter1,
                          kernel_size=3, batch_norm=batch_norm)
        self.conv2 = Conv(n_filter1, n_filter1,
                          kernel_size=3, batch_norm=batch_norm)
        self.conv3 = Conv(n_filter1, n_filter2, kernel_size=3,
                          stride=2, padding=1, batch_norm=batch_norm)

        self.dropout1 = self.features = nn.Sequential(
            nn.Dropout(inplace=True) if dropout else Identity())

        self.conv4 = Conv(n_filter2, n_filter2, kernel_size=3,
                          stride=1, batch_norm=batch_norm)
        self.conv5 = Conv(n_filter2, n_filter2, kernel_size=3,
                          stride=1, batch_norm=batch_norm)
        self.conv6 = Conv(n_filter2, n_filter2, kernel_size=3,
                          stride=2, padding=1, batch_norm=batch_norm)

        self.dropout2 = self.features = nn.Sequential(
            nn.Dropout(inplace=True) if dropout else Identity())

        self.conv7 = Conv(n_filter2, n_filter2, kernel_size=3,
                          stride=1, batch_norm=batch_norm)
        self.conv8 = Conv(n_filter2, n_filter2, kernel_size=1,
                          stride=1, batch_norm=batch_norm)
        # self.pool = nn.AvgPool2d(7)
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.flatten = Flatten()

        self.classifier = nn.Sequential(
            nn.Linear(n_filter2*2*2, num_classes),
        )

    def forward(self, x):
        out = self.conv1(x)

        out = self.conv2(out)

        out = self.conv3(out)

        out = self.dropout1(out)

        out = self.conv4(out)

        out = self.conv5(out)

        out = self.conv6(out)

        out = self.dropout2(out)

        out = self.conv7(out)

        out = self.conv8(out)

        out = self.pool(out)

        out = self.flatten(out)

        out = self.classifier(out)

        return out


class ResNet18(nn.Module):
    def __init__(self, num_channels=3, num_classes=10, pretrained=False, dataset='cifar10'):
        super().__init__()
        
        # 1. 加载基础模型
        if pretrained:
            from torchvision.models import ResNet18_Weights
            try:
                base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            except:
                base = resnet18(pretrained=True)
        else:
            base = resnet18(weights=None)

        # 2. [动态修复] 根据数据集调整 Conv1 结构
        # TinyImageNet(64x64) 的 checkpoint 使用的是标准 7x7 卷积 ([64, 3, 7, 7])
        # CIFAR(32x32) / MNIST(28x28) 使用的是修改版 3x3 卷积 ([64, c, 3, 3])
        # [修改] 如果开启预训练(pretrained=True)，CIFAR也应该保留原生结构(7x7)以匹配权重
        # 只有 MNIST(通道不对) 或者 不使用预训练的CIFAR(为了适应小图) 才替换 conv1
        if dataset == 'mnist' or (not pretrained and dataset in ['cifar10', 'cifar100']):
            base.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
            base.maxpool = nn.Identity()
        else:
            # TinyImageNet 或其他：保持标准 ResNet 结构 (7x7, stride=2)
            # 或者是开启了预训练的 CIFAR (此时我们将在 data_utils 里把图片放大)
            # 如果通道数不是3 (虽然TinyImageNet是3)，需重置第一层以匹配通道
            if num_channels != 3:
                base.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 3. 截取全连接层前的部分
        self.base = nn.Sequential(*list(base.children())[:-1])

        
        # 4. 修改全连接层
        in_features = base.fc.in_features
        self.drop = nn.Dropout()
        self.final = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.base(x)
        x = self.drop(x.view(-1, self.final.in_features))
        x = self.final(x)
        return x
        
class ResNet50(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        base = resnet50(pretrained=pretrained)
        self.base = nn.Sequential(*list(base.children())[:-1])
        if pretrained:
            for param in self.base.parameters():
                param.requires_grad = False
        in_features = base.fc.in_features
        self.drop = nn.Dropout()
        self.final = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.base(x)
        x = self.drop(x.view(-1, self.final.in_features))
        x = self.final(x)
        return x
