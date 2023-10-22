import warnings
import torch
import random
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import time
import sys
import numpy as np
import os

try:
    sys.path.insert(0, '.')
    from model.base import Model
    from dataset.base import Dataset
    from metric import metrics
    from optimizer import Optimizer
    from scheduler import Scheduler, print_scheduler
    from tools import parse, get_logger, print_dct
finally:
    pass

warnings.filterwarnings('ignore')


def train():
    args = parse()
    args_str = print_dct(args)

    torch.manual_seed(args['train']['seed'])
    random.seed(args['train']['seed'])

    epochs = args['train']['epochs']
    bs = args['dataset']['batch_size']
    device_str = args['train']['device']
    device = torch.device(args['train']['device'])
    se = args['train']['save_every']

    optimizer_name = args['optimizer']['name']
    optimizer_params = args['optimizer']['parameters']

    model = Model(args['model'])

    optimizer = Optimizer(optimizer_name, optimizer_params, model.parameters())
    scheduler = Scheduler(optimizer, args['scheduler'])

    loss_fn = CrossEntropyLoss()

    if 'checkpoint' in args['train'].keys():
        model.load_state_dict(torch.load(args['train']['checkpoint']))

    log_path = f"{args['train']['log_path']}{args['model']['name'].lower()}/"
    if not os.path.exists(log_path):
        os.system(f'mkdir -p {log_path}')
    logger = get_logger(log_path + 'train.log')

    save_path = f"{args['train']['save_path']}{args['model']['name'].lower()}/"
    if not os.path.exists(save_path):
        os.system(f'mkdir -p {save_path}')

    model = model.to(device)

    logger.info(f'\n{args_str}')
    logger.info(f'\n{str(model)}')
    logger.info(f'\n{str(optimizer)}')
    scheduler_str = print_scheduler(scheduler, args['scheduler'])
    logger.info(scheduler_str)
    logger.info(f'\n{str(loss_fn)}')

    train_dataset, valid_dataset, _ = Dataset(args['dataset'])

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True
    )

    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=bs,
        shuffle=True
    )

    init_time = time.time()
    best_loss = 99999999.0

    epochs_npz = np.array([ep for ep in range(epochs)])
    train_loss_npz = []
    valid_loss_npz = []

    for epoch in range(epochs):
        start = time.time()
        model.train()

        train_loss = 0.0
        valid_loss = 0.0

        for batch in train_data_loader:
            imgs, labels = batch
            imgs = imgs.to(device)
            labels = labels.to(device)
            preds = model(imgs)
            loss = loss_fn(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            for batch in valid_data_loader:
                imgs, labels = batch
                imgs = imgs.to(device)
                labels = labels.to(device)
                preds = model(imgs)
                loss = loss_fn(preds, labels)
                valid_loss += loss.item()

        train_loss /= len(train_data_loader)
        valid_loss /= len(valid_data_loader)

        train_loss_npz.append(train_loss)
        valid_loss_npz.append(valid_loss)

        cond1 = valid_loss < best_loss
        cond2 = (epoch + 1) % se == 0
        cond3 = epoch == epochs - 1

        if cond1:
            best_loss = valid_loss

        if cond1 or cond2 or cond3:
            state_dict = model.state_dict()
            save_name = f'ep={epoch + 1}_lv={valid_loss:.2f}.pth'
            torch.save(state_dict, save_path + save_name)

        end = time.time()

        print_str = f'{device_str} '
        print_str += f'epoch: {epoch + 1}/{epochs} '
        print_str += f'train_loss: {train_loss:.4f} '
        print_str += f'valid_loss: {valid_loss:.4f} '
        print_str += f'epoch_time: {(end - start):.3f} sec'
        logger.info(print_str)

    last_time = time.time()

    train_metrics = metrics(model, train_data_loader, device)
    valid_metrics = metrics(model, valid_data_loader, device)

    print_str = '\n'

    for name, val in train_metrics:
        print_str += f'train_{name}: {val:.3f} '

    for name, val in valid_metrics:
        print_str += f'valid_{name}: {val:.3f} '

    print_str += f'total_time: {(last_time - init_time):.3f} sec'
    logger.info(print_str)

    np.savez_compressed(
        log_path + 'losses.npz',
        epochs_npz,
        train_loss_npz,
        valid_loss_npz
    )


if __name__ == '__main__':
    train()
