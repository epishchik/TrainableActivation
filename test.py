import warnings
import torch
import random
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import time
import sys
import os

try:
    sys.path.insert(0, '.')
    from model.base import Model
    from dataset.base import Dataset
    from metric import metrics
    from tools import parse, get_logger, print_dct
finally:
    pass

warnings.filterwarnings('ignore')


def test():
    args = parse()
    args_str = print_dct(args)

    torch.manual_seed(args['test']['seed'])
    random.seed(args['test']['seed'])

    bs = args['dataset']['batch_size']
    device_str = args['test']['device']
    device = torch.device(args['test']['device'])

    model = Model(args['model'])
    loss_fn = CrossEntropyLoss()

    ckpt_path = f"{args['test']['checkpoint']}{args['model']['name'].lower()}/"
    checkpoints = [ckpt_path + ckpt for ckpt in os.listdir(ckpt_path)]

    log_path = f"{args['test']['log_path']}{args['model']['name'].lower()}/"
    if not os.path.exists(log_path):
        os.system(f'mkdir -p {log_path}')
    logger = get_logger(log_path + 'test.log')

    model = model.to(device)

    logger.info(f'\n{args_str}')
    logger.info(f'\n{str(model)}')
    logger.info(f'\n{str(loss_fn)}')

    _, _, test_dataset = Dataset(args['dataset'])

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=bs,
        shuffle=False
    )

    start = time.time()

    for checkpoint in checkpoints:
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        model.eval()

        test_loss = 0.0

        with torch.no_grad():
            for batch in test_data_loader:
                imgs, labels = batch
                imgs = imgs.to(device)
                labels = labels.to(device)
                preds = model(imgs)
                loss = loss_fn(preds, labels)
                test_loss += loss.item()

        test_loss /= len(test_data_loader)

        print_str = f'{device_str} '
        print_str += f'ckpt: {checkpoint} '
        print_str += f'test_loss: {test_loss:.4f} '

        test_metrics = metrics(model, test_data_loader, device)
        for i, (name, metr) in enumerate(test_metrics):
            if i == len(test_metrics) - 1:
                print_str += f'test_{name}: {metr:.3f}'
            else:
                print_str += f'test_{name}: {metr:.3f} '

        logger.info(print_str)

    end = time.time()

    print_str = f'total_time: {(end - start):.3f} sec'
    logger.info(print_str)


if __name__ == '__main__':
    test()
