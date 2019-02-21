import os
import random
import glob
import numpy as np
from keras.utils import to_categorical
from keras.models import Model

from utility.utility import project_root, retrieve_classes

project_root = project_root()


def create_sequence_generator(
        subset, classes, batch_size, feature_sequence_length, feature_length, flow_feature, number_of_segment):
    if batch_size % number_of_segment != 0:
        raise Exception('Batch size must be divisible by number of segment')

    counter = 0

    if flow_feature:
        folder_suffix = '_flowfeatures'
        dataset_folder_name = 'UCF-101-flow-features'
    else:
        folder_suffix = '_features'
        dataset_folder_name = 'UCF-101-rgb-features'

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
            feature_filename_pattern = os.path.join(project_root, 'data', dataset_folder_name,
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
        subset, batch_size, feature_sequence_length, feature_length, model_flow, model_rgb, number_of_segment):

    classes = retrieve_classes()
    counter = 0

    model_flow = Model(
        inputs=model_flow.input,
        outputs=model_flow.get_layer('flow_dense2').output
    )

    model_rgb = Model(
        inputs=model_rgb.input,
        outputs=model_rgb.get_layer('rgb_dense2').output
    )
    model_flow._make_predict_function()
    model_rgb._make_predict_function()

    model_rgb.summary()
    model_flow.summary()

    flow_folder_suffix = '_flowfeatures'
    flow_dataset_folder_name = 'UCF-101-flow-features'
    rgb_folder_suffix = '_features'
    rgb_dataset_folder_name = 'UCF-101-rgb-features'

    while True:  # This is because generator must be infinite loop
        x_batch_rgb = []
        x_batch_flow = []
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
            rgb_feature_filename_pattern = os.path.join(project_root, 'data',
                                                    rgb_dataset_folder_name, video.label, video.label + rgb_folder_suffix,
                                                    'v_' + video.label + '_' + video.group + '_' + video.clip + '*.npy')

            flow_feature_filename_pattern = os.path.join(project_root, 'data',
                                                    flow_dataset_folder_name, video.label, video.label + flow_folder_suffix,
                                                    'v_' + video.label + '_' + video.group + '_' + video.clip + '*.npy')

            rgb_segment_features_list = glob.glob(rgb_feature_filename_pattern)
            flow_segment_features_list = glob.glob(flow_feature_filename_pattern)

            if len(rgb_segment_features_list) != number_of_segment:
                raise Exception('A video has a wrong number of feature part. Please check your parameters')
            if len(flow_segment_features_list) != number_of_segment:
                raise Exception('A video has a wrong number of feature part. Please check your parameters')

            predictions_list_rgb = []
            predictions_list_flow = []
            for rgb_segment_feature in rgb_segment_features_list:

                if not os.path.isfile(rgb_segment_feature):
                    raise Exception('Feature not found! You have to create all the features!')

                # Retrieving features
                features_sequence = np.load(rgb_segment_feature)
                features_sequence = features_sequence.reshape((1, feature_sequence_length, feature_length))
                prediction_rgb = model_rgb.predict(features_sequence, batch_size=1, verbose=1)
                predictions_list_rgb.append(prediction_rgb)

            for flow_segment_feature in flow_segment_features_list:

                if not os.path.isfile(flow_segment_feature):
                    raise Exception('Feature not found! You have to create all the features!')

                # Retrieving features
                features_sequence = np.load(flow_segment_feature)
                features_sequence = features_sequence.reshape((1, feature_sequence_length, feature_length))
                prediction_flow = model_flow.predict(features_sequence, batch_size=1, verbose=1)
                predictions_list_flow.append(prediction_flow)

            x_batch_rgb.append(predictions_list_rgb)
            x_batch_flow.append(predictions_list_flow)
            y_batch.append(to_categorical(classes.index(video.label), len(classes)))

        yield [np.array(x_batch_rgb), np.array(x_batch_flow)], np.array(y_batch)
