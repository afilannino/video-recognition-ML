import matplotlib.pyplot as plt
import os
import pandas as pd

from utility.utility import project_root

project_root = project_root()
save_pic = True


def main(log_file_to_plot='training-1541372600.5267973.log'):
    print('Utility function')
    epoch, acc, loss, topk = retrieve_plot_data(log_file_to_plot, 'train')
    epoch, val_acc, val_loss, val_topk = retrieve_plot_data(log_file_to_plot, 'validation')
    create_plot(epoch, acc, val_acc, 'Accuracy')
    create_plot(epoch, loss, val_loss, 'Loss')
    create_plot(epoch, topk, val_topk, 'TopK')
    print('END!')


def retrieve_plot_data(filelog_name, phase):
    log_path = os.path.join(project_root, 'data', 'result', 'logs', filelog_name)
    if not os.path.exists(log_path):
        raise Exception('Log file not found!')

    log_data = pd.read_csv(log_path, delimiter=',')
    epoch = log_data.epoch.tolist()

    if phase == 'train':
        acc = log_data.acc.tolist()
        loss = log_data.loss.tolist()
        top_k_categorical_accuracy = log_data.top_k_categorical_accuracy.tolist()
    elif phase == 'validation':
        acc = log_data.val_acc.tolist()
        loss = log_data.val_loss.tolist()
        top_k_categorical_accuracy = log_data.val_top_k_categorical_accuracy.tolist()
    else:
        acc = []
        loss = []
        top_k_categorical_accuracy = []

    return epoch, acc, loss, top_k_categorical_accuracy


def create_plot(epoch, train_values, validation_values=None, metric='Metric'):
    fig, ax = plt.subplots()
    ax.plot(epoch, train_values, label='train')

    if validation_values is not None:
        ax.plot(epoch, validation_values, label='validation')

    ax.set(xlabel='Epoch', ylabel=metric, title=metric + ' over epochs')
    ax.grid()
    ax.legend()
    # plt.show()

    if save_pic:
        save_path = os.path.join(project_root, 'data', 'result')
        fig.savefig(os.path.join(save_path, metric + '.png'))


if __name__ == '__main__':
    main()
