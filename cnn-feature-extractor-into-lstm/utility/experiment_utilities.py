import os
import random
import numpy as np
from keras.utils import to_categorical

from utility.utility import project_root

project_root = project_root()


def retrieve_sequence(subset, classes):
    x_subset = []
    y_subset = []
    for video in subset:
        features_folder_name = os.path.join(project_root, 'data', 'UCF-101', video.label, video.label + '_features')
        feature_filename = os.path.join(features_folder_name,
                                        'v_' + video.label + '_' + video.group + '_' + video.clip + '.npy')
        if os.path.isfile(feature_filename):
            features_sequence = np.load(feature_filename)
        else:
            raise Exception('Feature not found! You have to create all the features!')

        x_subset.append(features_sequence)
        y_subset.append(to_categorical(classes.index(video.label), len(classes)))

    return np.array(x_subset), np.array(y_subset)


def create_sequence_generator(subset, classes, batch_size):
    counter = 0
    while True:  # This is because generator must be infinite loop in Keras
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
            for i in range(batch_size - missing):
                current_batch.append(subset[i])
            counter = i + 1

        if counter == len(subset):
            counter = 0
            random.shuffle(subset)

        for video in current_batch:
            feature_filename = os.path.join(project_root, 'data', 'UCF-101', video.label, video.label + '_features',
                                            'v_' + video.label + '_' + video.group + '_' + video.clip + '.npy')
            if os.path.isfile(feature_filename):
                features_sequence = np.load(feature_filename)
            else:
                raise Exception('Feature not found! You have to create all the features!')

            x_batch.append(features_sequence)
            y_batch.append(to_categorical(classes.index(video.label), len(classes)))

        yield np.array(x_batch), np.array(y_batch)
