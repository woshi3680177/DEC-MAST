import datetime
import os
import matplotlib.pyplot as plt
import argparse
import time
import numpy as np
import tensorflow as tf
from keras import layers, losses, Model, optimizers, regularizers
from keras.layers import Input, Flatten, Dense, Reshape, Conv2DTranspose, Conv2D, Concatenate
from sklearn.cluster import KMeans
from keras.losses import categorical_crossentropy
from utils import get_xy, log_csv_reselect, get_ACC_NMI_ARI
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from scipy.spatial.distance import cdist

def model_conv():
    init = 'uniform'
    filters = [32, 64, 128, hidden_units]
    if input_shape[0] % 8 == 0:
        pad3 = 'same'
    else:
        pad3 = 'valid'
    input = Input(shape=input_shape)
    x = Conv2D(filters[0], kernel_size=5, strides=2, padding='same', activation='relu', kernel_initializer=init)(input)
    x = Conv2D(filters[1], kernel_size=5, strides=2, padding='same', activation='relu', kernel_initializer=init)(x)
    x = Conv2D(filters[2], kernel_size=3, strides=2, padding=pad3, activation='relu', kernel_initializer=init)(x)
    x = Flatten()(x)
    x = Dense(units=filters[-1], name='embed')(x)
    h = x
    x = Dense(filters[2] * (input_shape[0] // 8) * (input_shape[0] // 8), activation='relu')(
        x)
    x = Reshape((input_shape[0] // 8, input_shape[0] // 8, filters[2]))(x)
    x = Conv2DTranspose(filters[1], kernel_size=3, strides=2, padding=pad3, activation='relu')(x)
    x = Conv2DTranspose(filters[0], kernel_size=5, strides=2, padding='same', activation='relu')(x)
    x = Conv2DTranspose(input_shape[2], kernel_size=5, strides=2, padding='same')(x)
    output = Concatenate()([h, Flatten()(x)])
    model = Model(inputs=input, outputs=output)
    # model.summary()
    return model

def reselect_conv(input_shape):
    init = 'uniform'
    filters = [32, 64, 128, 10]
    if input_shape[0] % 8 == 0:
        pad3 = 'same'
    else:
        pad3 = 'valid'
    input_tensor = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(filters[0], kernel_size=5, strides=2, padding='same', activation='relu', kernel_initializer=init)(
        input_tensor)
    x = layers.Conv2D(filters[1], kernel_size=5, strides=2, padding='same', activation='relu', kernel_initializer=init)(
        x)
    x = layers.Conv2D(filters[2], kernel_size=3, strides=2, padding=pad3, activation='relu', kernel_initializer=init)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(units=filters[-1], name='embed')(x)
    encoder_output = x
    x = layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.01))(encoder_output)
    x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(10, activation='softmax', name='predictions')(x)
    reselect_conv = tf.keras.Model(input_tensor, output, name='reselect_model')
    # reselect_conv.summary()
    return reselect_conv


def load_weights_to_reselect(model_conv, reselect_conv):
    model_conv.load_weights(args.save_dir + '/FASHION.h5')
    model_conv_encoder_weights = model_conv.get_layer('embed').get_weights()
    reselect_conv.get_layer('embed').set_weights(model_conv_encoder_weights)


def compute_euclidean_distance(samples, cluster_centers):
    distances = np.sqrt(np.sum((samples[:, np.newaxis] - cluster_centers) ** 2, axis=2))
    return distances


def adaptive_datagen(rotation_range, width_shift_range, height_shift_range, zoom_range):
    num_samples = len(x)
    if num_samples < 1500:
        rotation_range *= 2
        width_shift_range *= 2
        height_shift_range *= 2
    datagen = ImageDataGenerator(
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        shear_range=0.2,
        zoom_range=zoom_range,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    return datagen


def plot_histogram(min_distances):
    plt.figure(figsize=(30, 15))
    n, bins, _ = plt.hist(min_distances, bins=10, color='royalblue')
    print("bins:", bins)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    print("bin_centers:", bin_centers)
    plt.plot(bin_centers, n, c='r', ls='-', marker='*', markersize=18, linewidth=6, alpha=0.7)
    plt.xlabel("Distance")
    plt.ylabel("Number of samples")
    plt.title("Distribution of distances on the Fashion MNIST dataset")
    plt.grid(True, alpha=0.5)
    plt.tick_params(labelsize=48)
    plt.show()

def train(x, y):
    log_str_re = f'iter; acc; nmi; ari; time:{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}'
    log_csv_reselect(log_str_re.split(';'), file_name=ds_name)
    model = model_conv()
    model.load_weights(args.save_dir + '/FASHION.h5')
    kmeans_n_init = 140
    assignment = np.array([-1] * len(x))
    for ite in range(15):
        H = model(x).numpy()[:, :hidden_units]
        ans_kmeans = KMeans(n_clusters=n_clusters, n_init=kmeans_n_init).fit(H)

        kmeans_n_init = int(ans_kmeans.n_iter_ * 2)

        U = ans_kmeans.cluster_centers_
        assignment_new = ans_kmeans.labels_
        w = np.zeros((n_clusters, n_clusters), dtype=np.int64)
        for i in range(len(assignment_new)):
            w[assignment_new[i], assignment[i]] += 1
        from scipy.optimize import linear_sum_assignment as linear_assignment
        ind = linear_assignment(-w)
        temp = np.array(assignment)
        for i in range(n_clusters):
            assignment[temp == ind[1][i]] = i
        n_change_assignment = np.sum(assignment_new != assignment)
        assignment = assignment_new

        distances = cdist(H, U, metric='euclidean')
        center_idx = np.argmin(distances, axis=0)
        new_centroids = H[center_idx, :]
        distances_s_c = compute_euclidean_distance(H, new_centroids)
        min_distances = np.min(distances_s_c, axis=1)

        # plot
        #plot_histogram(min_distances)

        import sys
        sys.stdout.flush()
        net = reselect_conv(input_shape=input_shape)
        load_weights_to_reselect(model_conv(), net)
        net.compile(optimizer=optimizers.Adam(learning_rate=1e-3), loss=categorical_crossentropy)

        # Data augmentation
        rotation_range = 15
        zoom_range = 0.01
        width_shift_range = 0.1
        height_shift_range = 0.1
        datagen = adaptive_datagen(rotation_range, width_shift_range, height_shift_range, zoom_range)
        datagen_batch_size = 64
        y_one_hot = to_categorical(y, num_classes=10)

        net.fit(datagen.flow(x, y_one_hot, batch_size=datagen_batch_size), epochs=50)

        # multiple threshold voting
        thresholds = [0.125, 0.375, 0.625, 0.875]
        threshold_results = {}
        for threshold in thresholds:
            binary_result = (min_distances <= threshold).astype(int)
            threshold_results[threshold] = binary_result
        sample_count = len(min_distances)
        reliable_counts = np.zeros(sample_count)

        for threshold, result_matrix in threshold_results.items():
            reliable_counts += result_matrix
        final_reliable_samples = (reliable_counts >= len(thresholds) / 2).astype(int)

        reliable_samples_indices = np.where(final_reliable_samples == 1)[0]
        final_reliable_samples = np.array([x[i] for i in reliable_samples_indices])
        final_reliable_samples_labels = to_categorical([assignment[i] for i in reliable_samples_indices])

        from sklearn.utils import shuffle
        trusted_samples_shuffled, trusted_labels_shuffled = shuffle(final_reliable_samples,
                                                                    final_reliable_samples_labels)

        steps_per_epoch = np.ceil(len(trusted_samples_shuffled) / batch_size)

        net.fit(
            datagen.flow(trusted_samples_shuffled, trusted_labels_shuffled, batch_size=batch_size),
            steps_per_epoch=steps_per_epoch,
            epochs=50,
            verbose=2
        )

        predictions = net.predict(x)

        predictions = [np.argmax(l) for l in predictions]

        predictions = np.array(predictions)

        acc_re, nmi_re, ari_re = get_ACC_NMI_ARI(np.array(y), np.array(predictions))

        log_str_re = f'{ite}; {acc_re}; {nmi_re}; {ari_re}; acc, nmi,ari = {acc_re, nmi_re, ari_re}; ' \
                     f'time:{time.time() - time_start:.3f}'

        print("The clustering results after the reselect network ===========>")

        print(log_str_re)

        log_csv_reselect(log_str_re.split(';'), file_name=ds_name)

        if not os.path.exists("weights/Fashion_MNIST"):
            os.makedirs("weights/Fashion_MNIST")

        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = f"weights/Fashion_MNIST/reselect_weight_final_{ds_name}_{current_time}_vote_result.h5"

        net.save_weights(filename)


if __name__ == '__main__':

    batch_size = 128
    hidden_units = 10

    parser = argparse.ArgumentParser(description='select dataset:MNIST,COIL20,USPS,FASHION')
    parser.add_argument('ds_name', default='MNIST')
    parser.add_argument('--save_dir', default='results/IDECT')
    parser.add_argument('--threshold', type=float, default=0.6, help='Threshold for selecting reliable samples')
    args = parser.parse_args()
    args.save_dir = f'results/IDECT'

    if args.ds_name is None or not args.ds_name in ['MNIST',  'COIL20', 'USPS', 'FASHION']:
        ds_name = 'MNIST'
    else:
        ds_name = args.ds_name

    if ds_name == 'MNIST':
        input_shape = (28, 28, 1)
        n_clusters = 10

    elif ds_name == 'USPS':
        input_shape = (16, 16, 1)
        n_clusters = 10

    elif ds_name == 'COIL20':
        input_shape = (28, 28, 1)
        n_clusters = 20

    elif ds_name == 'FASHION':
        input_shape = (28, 28, 1)
        n_clusters = 10

    time_start = time.time()

    x, y = get_xy(ds_name=ds_name)

    train(x, y)

    print(time.time() - time_start)
