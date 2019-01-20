import os
import random
import glob
import numpy as np
from keras.utils import to_categorical

from utility.utility import project_root, retrieve_classes

project_root = project_root()


def create_sequence_generator(
        subset, classes, batch_size, feature_sequence_length, feature_length, flow_feature, number_of_segment):
    if batch_size % number_of_segment != 0:
        raise Exception('Batch size must be divisible by number of segment')

    counter = 0

    if flow_feature:
        folder_suffix = '_flowfeatures'
    else:
        folder_suffix = '_features'

    while True:  # This is because generator must be infinite loop
        x_batch = []
        y_batch = []
        current_batch = []

        if counter + batch_size / number_of_segment <= len(subset):
            for i in range(batch_size // number_of_segment):
                current_batch.append(subset[i + counter])
            counter += batch_size // number_of_segment

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
            feature_filename_pattern = os.path.join(project_root, 'data', 'UCF-101',
                                                    video.label, video.label + folder_suffix,
                                                    'v_' + video.label + '_' + video.group + '_' + video.clip + '*.npy')

            segment_features_list = glob.glob(feature_filename_pattern)

            if len(segment_features_list) != number_of_segment:
                raise Exception('A video has a wrong number of feature part. Please check your parameters')

            for segment_feature in segment_features_list:

                if not os.path.isfile(segment_feature):
                    raise Exception('Feature not found! You have to create all the features!')

                # Retrieving features
                features_sequence = np.load(segment_feature)
                length = len(features_sequence)
                if length > feature_sequence_length:
                    raise Exception('Some sequences of features are too long! Please compute them again!')
                # Zero padding if size is different from size_limit
                elif length < feature_sequence_length:
                    features_sequence = np.append(features_sequence,
                                                  np.zeros((feature_sequence_length - length, feature_length),
                                                           dtype='float32'))
                    features_sequence = features_sequence.reshape((feature_sequence_length, feature_length))

                x_batch.append(features_sequence)
                y_batch.append(to_categorical(classes.index(video.label), len(classes)))

        yield np.array(x_batch), np.array(y_batch)


def create_sequence_of_tuple_generator(
        subset, batch_size, feature_sequence_length, feature_length, model, flow_feature, number_of_segment):

    classes = retrieve_classes()
    counter = 0

    if flow_feature:
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
            feature_filename_pattern = os.path.join(project_root, 'data',
                                                    'UCF-101', video.label, video.label + folder_suffix,
                                                    'v_' + video.label + '_' + video.group + '_' + video.clip + '*.npy')

            segment_features_list = glob.glob(feature_filename_pattern)

            if len(segment_features_list) != number_of_segment:
                raise Exception('A video has a wrong number of feature part. Please check your parameters')

            predictions_list = []
            for segment_feature in segment_features_list:

                if not os.path.isfile(segment_feature):
                    raise Exception('Feature not found! You have to create all the features!')

                # Retrieving features
                features_sequence = np.load(segment_feature)
                length = len(features_sequence)
                if length > feature_sequence_length:
                    raise Exception('Some sequences of features are too long! Please compute them again!')
                # Zero padding if size is different from size_limit
                elif length < feature_sequence_length:
                    features_sequence = np.append(features_sequence,
                                                  np.zeros((feature_sequence_length - length, feature_length),
                                                           dtype='float32'))
                    features_sequence = features_sequence.reshape((feature_sequence_length, feature_length))

                prediction = model.predict(features_sequence, batch_size=1, verbose=1)
                predictions_list.append(prediction)

            x_batch.append(predictions_list)
            y_batch.append(to_categorical(classes.index(video.label), len(classes)))

        yield np.array(x_batch), np.array(y_batch)
