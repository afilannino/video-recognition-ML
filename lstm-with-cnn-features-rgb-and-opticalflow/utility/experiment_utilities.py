import os
import random
import csv
import numpy as np
from keras.utils import to_categorical

from utility.utility import project_root

project_root = project_root()


def retrieve_sequence(subset, classes, feature_sequence_length, feature_length, flow_features=False):
    x_subset = []
    y_subset = []

    if flow_features:
        folder_suffix = '_flowfeatures'
    else:
        folder_suffix = '_features'

    for video in subset:
        features_folder_name = os.path.join(project_root, 'data', 'UCF-101', video.label, video.label + folder_suffix)
        feature_filename = os.path.join(features_folder_name,
                                        'v_' + video.label + '_' + video.group + '_' + video.clip + '.npy')
        if os.path.isfile(feature_filename):
            features_sequence = np.load(feature_filename)
            length = len(features_sequence)
            if length > feature_sequence_length:
                raise Exception('Some sequences of features are too long! Please compute them again!')
            # Zero padding if size is different from size_limit
            elif length < feature_sequence_length:
                features_sequence = np.append(features_sequence, np.zeros((feature_sequence_length - length,
                                                                           feature_length), dtype='float32'))
                features_sequence = features_sequence.reshape((feature_sequence_length, feature_length))
        else:
            raise Exception('Feature not found! You have to create all the features!')

        x_subset.append(features_sequence)
        y_subset.append(to_categorical(classes.index(video.label), len(classes)))

    return np.array(x_subset), np.array(y_subset)


def create_sequence_generator(subset, classes, batch_size, feature_sequence_length, feature_length, flow_features=False):
    counter = 0
    if flow_features:
        folder_suffix = '_flowfeatures'
    else:
        folder_suffix = '_features'

    while True:  # This is because generator must be infinite loop
        x_batch = []
        y_batch = []
        current_batch = []

        if counter + batch_size <= len(subset):
            for i in range(batch_size):
                current_batch.append(subset[i + counter])
            counter += batch_size

        else:
            missing = len(subset) - counter
            for i in range(missing):
                current_batch.append(subset[i + counter])
            random.shuffle(subset)
            counter = 0

        if counter == len(subset):
            random.shuffle(subset)
            counter = 0

        for video in current_batch:
            feature_filename = os.path.join(project_root, 'data', 'UCF-101', video.label, video.label + folder_suffix,
                                            'v_' + video.label + '_' + video.group + '_' + video.clip + '.npy')
            if os.path.isfile(feature_filename):
                features_sequence = np.load(feature_filename)
                length = len(features_sequence)
                if length > feature_sequence_length:
                    raise Exception('Some sequences of features are too long! Please compute them again!')
                # Zero padding if size is different from size_limit
                elif length < feature_sequence_length:
                    features_sequence = np.append(features_sequence,
                                                  np.zeros((feature_sequence_length - length, feature_length),
                                                           dtype='float32'))
                    features_sequence = features_sequence.reshape((feature_sequence_length, feature_length))
            else:
                raise Exception('Feature not found! You have to create all the features!')

            x_batch.append(features_sequence)
            y_batch.append(to_categorical(classes.index(video.label), len(classes)))

        yield np.array(x_batch), np.array(y_batch)


def retrieve_weight():
    statistics_file = os.path.join(project_root, 'data', 'result', 'statistics.csv')
    with open(statistics_file, 'r') as csv_file:
        reader = csv.reader(csv_file)
        rows = [r for r in reader]
        weight = float(rows[10][1])
    return weight
