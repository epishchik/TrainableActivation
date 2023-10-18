from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
import sys
import torch

try:
    sys.path.insert(0, '../')
    import activation
finally:
    pass


def _weights_init_mnist(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


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


def _get_acts(train_act_acts):
    acts = []

    for act_name in train_act_acts:
        acts.append(getattr(F, act_name))

    return acts


class DNN(nn.Module):
    def __init__(self, features):
        super(DNN, self).__init__()

        self.in_features = 64
        self.out_features = 10
        self.prev_feature = features[0]

        self.first_fc = nn.Linear(self.in_features, features[0])
        self.first_act = nn.ReLU()

        self.num_features = len(features)
        if self.num_features > 1:
            self.layers = self._make_layers(features[1:])

        self.last_fc = nn.Linear(features[-1], self.out_features)
        self.apply(_weights_init_mnist)

    def _make_layers(self, features):
        layers = []

        for feature in features:
            layers.append(nn.Linear(self.prev_feature, feature))
            layers.append(nn.ReLU())
            self.prev_feature = feature

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.first_act(self.first_fc(x.view(x.size(0), -1)))
        if self.num_features > 1:
            out = self.layers(out)
        out = self.last_fc(out)
        return out


class DNN2(nn.Module):
    def __init__(self, params):
        super(DNN2, self).__init__()

        in_channels = params['in_channels']
        out_channels = params['out_channels']
        train_act = params['activation']

        features = [392]

        self.model = DNN(features)

        self.model.first_fc = nn.Linear(in_channels, features[0])
        self.model.last_fc = nn.Linear(features[-1], out_channels)

        replace_layers(train_act, self.model, nn.ReLU)

    def forward(self, x):
        out = self.model(x)
        return out

    def __str__(self):
        model_str = str(self.model)
        return model_str


class DNN3(nn.Module):
    def __init__(self, params):
        super(DNN3, self).__init__()

        in_channels = params['in_channels']
        out_channels = params['out_channels']
        train_act = params['activation']

        features = [392, 196]

        self.model = DNN(features)

        self.model.first_fc = nn.Linear(in_channels, features[0])
        self.model.last_fc = nn.Linear(features[-1], out_channels)

        replace_layers(train_act, self.model, nn.ReLU)

    def forward(self, x):
        out = self.model(x)
        return out

    def __str__(self):
        model_str = str(self.model)
        return model_str


class DNN5(nn.Module):
    def __init__(self, params):
        super(DNN5, self).__init__()

        in_channels = params['in_channels']
        out_channels = params['out_channels']
        train_act = params['activation']

        features = [512, 256, 128, 64]

        self.model = DNN(features)

        self.model.first_fc = nn.Linear(in_channels, features[0])
        self.model.last_fc = nn.Linear(features[-1], out_channels)

        replace_layers(train_act, self.model, nn.ReLU)

    def forward(self, x):
        out = self.model(x)
        return out

    def __str__(self):
        model_str = str(self.model)
        return model_str
