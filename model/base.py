import sys

try:
    sys.path.insert(0, '../')
    import model.resnet as resnet
    import model.dnn as dnn
finally:
    pass


def Model(params):
    name = params['name']
    architecture = params['architecture']

    if 'ResNet' in name:
        model = getattr(resnet, name)
    elif 'DNN' in name:
        model = getattr(dnn, name)

    return model(architecture)
