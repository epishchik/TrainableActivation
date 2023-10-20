import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

try:
    sys.path.insert(0, '.')
    from tools import parse, get_logger, print_dct
finally:
    pass

warnings.filterwarnings('ignore')


def _plot(
    ax,
    idx,
    dataset,
    log_path,
    x_label,
    y_label,
    model,
    activations,
    clip_min,
    clip_max,
    logger
):
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f'{dataset.lower()}')
    ax.grid(True)

    for act in activations:
        name = f'{log_path}{act.lower()}/{dataset.lower()}/{model.lower()}'
        name += '/losses.npz'

        try:
            npz = np.load(name)
        except Exception as _:
            continue

        epochs = npz['arr_0']
        train_loss = npz['arr_1']
        valid_loss = npz['arr_2']

        train_loss = np.clip(
            train_loss,
            a_min=clip_min,
            a_max=clip_max
        )

        valid_loss = np.clip(
            valid_loss,
            a_min=clip_min,
            a_max=clip_max
        )

        if idx % 2 == 0:
            ax.plot(
                epochs,
                train_loss,
                label=f'train_{act.lower()}'
            )
        else:
            ax.plot(
                epochs,
                valid_loss,
                label=f'valid_{act.lower()}'
            )
        if idx % 2 == 1:
            logger.info(
                f'processed {model.lower()} {dataset.lower()} {act.lower()}'
            )

    ax.legend()


def plot():
    args = parse()
    args_str = print_dct(args)

    num_subplots = len(args['datasets'])

    models = args['models']
    datasets = args['datasets']
    activations = args['activations']

    x_label = 'epoch'
    y_label = 'loss'

    clip_min, clip_max = args['clip']

    plot_path = args['plot_path']
    log_path = args['log_path']

    dpi = args['dpi']

    if not os.path.exists(plot_path):
        os.mkdir(plot_path)

    logger = get_logger(plot_path + 'plot.log')
    logger.info(args_str)

    for model in models:
        _, axes = plt.subplots(num_subplots, 2, figsize=(15, 5), dpi=dpi)

        plt.subplots_adjust(
            left=0.1,
            bottom=0.1,
            right=0.9,
            top=0.9,
            wspace=0.3,
            hspace=0.3
        )

        if num_subplots > 1:
            for i in range(len(axes)):
                for j in range(len(axes[i])):
                    _plot(
                        axes[i, j],
                        j,
                        datasets[i],
                        log_path,
                        x_label,
                        y_label,
                        model,
                        activations,
                        clip_min,
                        clip_max,
                        logger
                    )
        else:
            for i, ax in enumerate(axes):
                _plot(
                    ax,
                    i,
                    datasets[i // 2],
                    log_path,
                    x_label,
                    y_label,
                    model,
                    activations,
                    clip_min,
                    clip_max,
                    logger
                )

        plt.savefig(f'{plot_path}{model.lower()}.pdf', dpi=dpi)

        plt.cla()
        plt.clf()


if __name__ == '__main__':
    plot()
