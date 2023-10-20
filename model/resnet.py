from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
import sys

try:
    sys.path.insert(0, '../')
    import activation
finally:
    pass


def replace_layers(act, model, old):
    if isinstance(act, str):
        if act != 'Default':
            new = getattr(activation, act)
            _replace_layers(model, old, new())
    else:
        train_act_name = act['name']
        train_act_acts = act['activations']
        acts = _get_acts(train_act_acts)

        new = getattr(activation, train_act_name)
        _replace_layers(model, old, new(activations=acts))


def _replace_layers(model, old, new):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            _replace_layers(module, old, new)

        if isinstance(module, old):
            setattr(model, n, new)


def _weights_init_cifar(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


def _get_acts(train_act_acts):
    acts = []

    for act_name in train_act_acts:
        acts.append(getattr(F, act_name))

    return acts


class LambdaLayerCifar(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayerCifar, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlockCifar(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlockCifar, self).__init__()

        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        self.relu = nn.ReLU()

        if stride != 1 or in_planes != planes:
            if option == 'A':
                tup1 = (0, 0, 0, 0, planes // 4, planes // 4)
                def f(x): return F.pad(x[:, :, ::2, ::2], tup1, 'constant', 0)
                self.shortcut = LambdaLayerCifar(f)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNetCifar(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetCifar, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(
            3,
            16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        self.fc = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.apply(_weights_init_cifar)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class ResNet8(nn.Module):
    def __init__(self, params):
        super().__init__()

        in_channels = params['in_channels']
        out_channels = params['out_channels']

        train_act = params['activation']

        self.model = ResNetCifar(BasicBlockCifar, [1, 1, 1])

        self.model.conv1 = nn.Conv2d(
            in_channels,
            16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.model.fc = nn.Linear(64, out_channels, bias=True)
        replace_layers(train_act, self.model, nn.ReLU)

    def forward(self, x):
        out = self.model(x)
        return out

    def __str__(self):
        model_str = str(self.model)
        return model_str


class ResNet14(nn.Module):
    def __init__(self, params):
        super().__init__()

        in_channels = params['in_channels']
        out_channels = params['out_channels']

        train_act = params['activation']

        self.model = ResNetCifar(BasicBlockCifar, [2, 2, 2])

        self.model.conv1 = nn.Conv2d(
            in_channels,
            16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.model.fc = nn.Linear(64, out_channels, bias=True)
        replace_layers(train_act, self.model, nn.ReLU)

    def forward(self, x):
        out = self.model(x)
        return out

    def __str__(self):
        model_str = str(self.model)
        return model_str


class ResNet20(nn.Module):
    def __init__(self, params):
        super().__init__()

        in_channels = params['in_channels']
        out_channels = params['out_channels']

        train_act = params['activation']

        self.model = ResNetCifar(BasicBlockCifar, [3, 3, 3])

        self.model.conv1 = nn.Conv2d(
            in_channels,
            16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.model.fc = nn.Linear(64, out_channels, bias=True)
        replace_layers(train_act, self.model, nn.ReLU)

    def forward(self, x):
        out = self.model(x)
        return out

    def __str__(self):
        model_str = str(self.model)
        return model_str


class ResNet26(nn.Module):
    def __init__(self, params):
        super().__init__()

        in_channels = params['in_channels']
        out_channels = params['out_channels']

        train_act = params['activation']

        self.model = ResNetCifar(BasicBlockCifar, [4, 4, 4])

        self.model.conv1 = nn.Conv2d(
            in_channels,
            16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.model.fc = nn.Linear(64, out_channels, bias=True)
        replace_layers(train_act, self.model, nn.ReLU)

    def forward(self, x):
        out = self.model(x)
        return out

    def __str__(self):
        model_str = str(self.model)
        return model_str


class ResNet32(nn.Module):
    def __init__(self, params):
        super().__init__()

        in_channels = params['in_channels']
        out_channels = params['out_channels']

        train_act = params['activation']

        self.model = ResNetCifar(BasicBlockCifar, [5, 5, 5])

        self.model.conv1 = nn.Conv2d(
            in_channels,
            16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.model.fc = nn.Linear(64, out_channels, bias=True)
        replace_layers(train_act, self.model, nn.ReLU)

    def forward(self, x):
        out = self.model(x)
        return out

    def __str__(self):
        model_str = str(self.model)
        return model_str


class ResNet44(nn.Module):
    def __init__(self, params):
        super().__init__()

        in_channels = params['in_channels']
        out_channels = params['out_channels']

        train_act = params['activation']

        self.model = ResNetCifar(BasicBlockCifar, [7, 7, 7])

        self.model.conv1 = nn.Conv2d(
            in_channels,
            16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.model.fc = nn.Linear(64, out_channels, bias=True)
        replace_layers(train_act, self.model, nn.ReLU)

    def forward(self, x):
        out = self.model(x)
        return out

    def __str__(self):
        model_str = str(self.model)
        return model_str


class ResNet56(nn.Module):
    def __init__(self, params):
        super().__init__()

        in_channels = params['in_channels']
        out_channels = params['out_channels']

        train_act = params['activation']

        self.model = ResNetCifar(BasicBlockCifar, [9, 9, 9])

        self.model.conv1 = nn.Conv2d(
            in_channels,
            16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.model.fc = nn.Linear(64, out_channels, bias=True)
        replace_layers(train_act, self.model, nn.ReLU)

    def forward(self, x):
        out = self.model(x)
        return out

    def __str__(self):
        model_str = str(self.model)
        return model_str
