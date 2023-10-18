import sys

try:
    sys.path.insert(0, '../')
    import dataset.cifar as cifar
    import dataset.mnist as mnist
finally:
    pass


def Dataset(params):
    name = params['name']

    if 'CIFAR' in name:
        dataset = getattr(cifar, name)
    elif 'MNIST' in name:
        dataset = getattr(mnist, name)
    return dataset(params)
