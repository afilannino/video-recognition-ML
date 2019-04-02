import os
import random
import glob
import numpy as np
from keras.utils import to_categorical
from keras.models import Model

from utility.utility import project_root, retrieve_classes

project_root = project_root()


def create_sequence_generator(
        subset, classes, batch_size, feature_sequence_length, feature_length, number_of_segment):
    if batch_size % number_of_segment != 0:
        raise Exception('Batch size must be divisible by number of segment')

    counter = 0

    flow_folder_suffix = '_flowfeatures'
    flow_dataset_folder_name = 'UCF-101-flow-features'
    rgb_folder_suffix = '_features'
    rgb_dataset_folder_name = 'UCF-101-rgb-features'

    while True:  # This is because generator must be infinite loop
        rgb_x_batch = []
        rgb_y_batch = []
        flow_x_batch = []
        flow_y_batch = []
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
            flow_feature_filename_pattern = os.path.join(project_root, 'data', flow_dataset_folder_name,
                                                    video.label, video.label + flow_folder_suffix,
                                                    'v_' + video.label + '_' + video.group + '_' + video.clip + '*.npy')
            rgb_feature_filename_pattern = os.path.join(project_root, 'data', rgb_dataset_folder_name,
                                                    video.label, video.label + rgb_folder_suffix,
                                                    'v_' + video.label + '_' + video.group + '_' + video.clip + '*.npy')

            rgb_segment_features_list = glob.glob(rgb_feature_filename_pattern)
            flow_segment_features_list = glob.glob(flow_feature_filename_pattern)

            if len(rgb_segment_features_list) != number_of_segment or len(flow_segment_features_list) != number_of_segment:
                raise Exception('A video has a wrong number of feature parts. Please check your parameters')

            for segment_feature in rgb_segment_features_list:
                if not os.path.isfile(segment_feature):
                    raise Exception('Feature not found! You have to create all the features!')
                features_sequence = load_and_pad(segment_feature, feature_sequence_length, feature_length)
                rgb_x_batch.append(features_sequence)
                rgb_y_batch.append(to_categorical(classes.index(video.label), len(classes)))

            for segment_feature in flow_segment_features_list:
                if not os.path.isfile(segment_feature):
                    raise Exception('Feature not found! You have to create all the features!')
                features_sequence = load_and_pad(segment_feature, feature_sequence_length, feature_length)
                flow_x_batch.append(features_sequence)
                flow_y_batch.append(to_categorical(classes.index(video.label), len(classes)))

        yield [np.array(rgb_x_batch), np.array(flow_x_batch)], [np.array(rgb_y_batch), np.array(flow_y_batch)]


def load_and_pad(feature_name, feature_sequence_length, feature_length):
    # Retrieving features
    features_sequence = np.load(feature_name)
    length = len(features_sequence)
    if length > feature_sequence_length:
        raise Exception('Some sequences of features are too long! Please compute them again!')
    # Zero padding if size is different from size_limit
    elif length < feature_sequence_length:
        features_sequence = np.append(features_sequence,
                                      np.zeros((feature_sequence_length - length, feature_length),
                                               dtype='float32'))
        features_sequence = features_sequence.reshape((feature_sequence_length, feature_length))

    return features_sequence


def create_sequence_generator_for_v2(
        subset, classes, batch_size, feature_sequence_length, feature_length, number_of_segment):
    if batch_size % number_of_segment != 0:
        raise Exception('Batch size must be divisible by number of segment')

    counter = 0

    flow_folder_suffix = '_flowfeatures'
    flow_dataset_folder_name = 'UCF-101-flow-features'
    rgb_folder_suffix = '_features'
    rgb_dataset_folder_name = 'UCF-101-rgb-features'

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
            flow_feature_filename_pattern = os.path.join(project_root, 'data', flow_dataset_folder_name,
                                                    video.label, video.label + flow_folder_suffix,
                                                    'v_' + video.label + '_' + video.group + '_' + video.clip + '*.npy')
            rgb_feature_filename_pattern = os.path.join(project_root, 'data', rgb_dataset_folder_name,
                                                    video.label, video.label + rgb_folder_suffix,
                                                    'v_' + video.label + '_' + video.group + '_' + video.clip + '*.npy')

            rgb_segment_features_list = glob.glob(rgb_feature_filename_pattern)
            flow_segment_features_list = glob.glob(flow_feature_filename_pattern)

            if len(rgb_segment_features_list) != number_of_segment or len(flow_segment_features_list) != number_of_segment:
                raise Exception('A video has a wrong number of feature parts. Please check your parameters')

            for i in range(len(rgb_segment_features_list)):

                rgb_segment_feature = rgb_segment_features_list[i]
                if not os.path.isfile(rgb_segment_feature):
                    raise Exception('Feature not found! You have to create all the features!')
                rgb_features_sequence = load_and_pad(rgb_segment_feature, feature_sequence_length, feature_length)

                flow_segment_feature = flow_segment_features_list[i]
                if not os.path.isfile(flow_segment_feature):
                    raise Exception('Feature not found! You have to create all the features!')
                flow_features_sequence = load_and_pad(flow_segment_feature, feature_sequence_length, feature_length)

                features_concatenation = np.concatenate((rgb_features_sequence, flow_features_sequence))
                x_batch.append(np.array(features_concatenation))
                y_batch.append(to_categorical(classes.index(video.label), len(classes)))

        yield np.array(x_batch), np.array(y_batch)
