from argparse import ArgumentParser
import yaml
import logging
import sys


def parse():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='./config.yaml')
    args = parser.parse_args()

    with open(args.config) as stream:
        data = yaml.safe_load(stream)
    return data


def get_logger(save_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(save_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    return logger


def _print_dct(dct, indent, indent_val):
    res_str = ''

    for key, value in dct.items():
        if isinstance(value, dict):
            res_str += f'{indent}{key}:\n'
            res_str += _print_dct(value, indent + indent_val, indent_val)
        else:
            res_str += indent + f'{key}: {value}\n'

    return res_str


def print_dct(dct, indent='', indent_val=2 * ' '):
    res_str = _print_dct(dct, indent, indent_val)

    cnt = 0

    for el in res_str[::-1]:
        if el == '\n':
            cnt += 1
        else:
            break

    if cnt != 0:
        return res_str[:-cnt]
    else:
        return res_str
