import csv
import os
import h5py
import numpy as np
import tensorflow as tf
from keras.datasets import fashion_mnist, mnist

def get_ACC_NMI_ARI(_y, _y_pred):
    y = np.array(_y)
    y_pred = np.array(_y_pred)
    s = np.unique(y_pred)
    t = np.unique(y)

    N = len(np.unique(y_pred))
    C = np.zeros((N, N), dtype=np.int32)
    for i in range(N):
        for j in range(N):
            idx = np.logical_and(y_pred == s[i], y == t[j])
            C[i][j] = np.count_nonzero(idx)
    Cmax = np.amax(C)
    C = Cmax - C
    from scipy.optimize import linear_sum_assignment
    row, col = linear_sum_assignment(C)
    count = 0
    for i in range(N):
        idx = np.logical_and(y_pred == s[row[i]], y == t[col[i]])
        count += np.count_nonzero(idx)
    acc = np.round(1.0 * count / len(y), 5)
    temp = np.array(y_pred)
    for i in range(N):
        y_pred[temp == col[i]] = i
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    nmi = np.round(normalized_mutual_info_score(y, y_pred), 5)
    ari = np.round(adjusted_rand_score(y, y_pred), 5)
    return acc, nmi, ari

def get_xy(ds_name='REUTERS', dir_path=r'datasets/', log_print=True, shuffle_seed=None):
    dir_path = dir_path + ds_name + '/'
    if ds_name == 'COIL20':
        f = h5py.File(dir_path + 'COIL20.h5', 'r')
        x = np.array(f['data'][()]).squeeze()
        x = np.expand_dims(np.swapaxes(x, 1, 2).astype(np.float32), -1)
        x = tf.image.resize(x, [28, 28]).numpy()
        x = x / 255.
        y = np.array(f['labels'][()]).astype(np.float32)
        y[y == 20.] = 0.

    elif ds_name == 'USPS':
        with h5py.File(dir_path + 'USPS.h5', 'r') as hf:
            train = hf.get('train')
            X_tr = train.get('data')[:]
            y_tr = train.get('target')[:]
            test = hf.get('test')
            X_te = test.get('data')[:]
            y_te = test.get('target')[:]
        x = np.concatenate([X_tr, X_te], 0)
        y = np.concatenate([y_tr, y_te], 0)
        x = np.reshape(x, (len(x), 16, 16)).astype(np.float32)

    elif ds_name == 'FASHION':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x = np.concatenate((x_train, x_test))
        y = np.concatenate((y_train, y_test))
        x = np.expand_dims(np.divide(x, 255.), -1)

    elif ds_name == 'MNIST':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x = np.concatenate((x_train, x_test))
        y = np.concatenate((y_train, y_test))
        x = np.expand_dims(np.divide(x, 255.), -1)

    if not shuffle_seed:
        shuffle_seed = int(np.random.randint(100))
    idx = np.arange(0, len(x))
    # tf.enable_eager_execution()
    idx = tf.random.shuffle(idx, seed=shuffle_seed).numpy()
    x = x[idx]
    y = y[idx]
    # x = tf.random.shuffle(x, seed=shuffle_seed).numpy()
    # y = tf.random.shuffle(y, seed=shuffle_seed).numpy()
    if log_print:
        print(ds_name)
    return x, y


def log_csv_reselect(strToWrite, file_name):
    path = r'log_history_reselect/'
    if not os.path.exists(path):
        os.makedirs(path)
    f = open(path + file_name + '.csv', 'a+', encoding='utf-8', newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(strToWrite)
    f.close()


def show_figure(data, label, shuffle=True, fileName=None):
    import matplotlib.pyplot as plt
    digit_size_S, digit_size_h, digit_size_w, channels = data.shape[0], data.shape[1], data.shape[2], data.shape[3]

    figure = np.zeros((digit_size_h * 10, digit_size_w * 20))
    if channels == 3:
        figure = np.zeros((digit_size_h * 10, digit_size_w * 20, channels))

    data = np.squeeze(data)
    if shuffle:
        import random
        idx = random.sample(range(0, data.shape[0]), 200)
    else:
        idx = range(200)
    print(label[idx])
    t = 0
    for i in range(10):
        for j in range(20):
            figure[i * digit_size_h: (i + 1) * digit_size_h, j * digit_size_w: (j + 1) * digit_size_w] = data[idx[t]]
            t = t + 1

    plt.figure(figsize=(10, 5))
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)

    plt.axis('off')
    plt.imshow(figure)

    if fileName is not None:
        plt.savefig('./picture/{}.png'.format(fileName))

    plt.show()


if __name__ == '__main__':
    # # load dataset:
    ds_name = 'MNIST_test'
    x, y = get_xy(ds_name=ds_name)
    print(np.min(x), np.max(x))
    print("x.shape={}, x.shape={}, 共{}类.".format(x.shape, y.shape, len(np.unique(y))))

    show_figure(x, y, shuffle=False, fileName=None)
    show_figure(x, y, shuffle=False, fileName=ds_name)