import torch.optim as optim


def Optimizer(name, params, model_params):
    optimizer = getattr(optim, name)
    return optimizer(model_params, **params)
