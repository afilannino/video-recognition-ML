import csv
import glob
import os.path
import time

import numpy as np
from keras.layers import Dense, Dropout, Input
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.optimizers import Adam

from utility.experiment_utilities import load_and_pad
from utility.utility import retrieve_videoobject_subsets, project_root, retrieve_classes

project_root = project_root()
feature_length = 2048  # This is the length of the numpy array that contains the prediction
classes_size = len(retrieve_classes())  # Number of classes in UCF101 dataset
feature_sequence_size = 10  # This must be coherent with the value used in features_extractor.py
number_of_segment = 3
classes = retrieve_classes()


def main():

    subsets = retrieve_videoobject_subsets(['validation'])

    # Create final model and retrieve training data
    model = create_model()

    # Validate model
    metrics = validate_model(model, subsets[1])
    save_result(metrics)


def create_model():
    input_shape = (feature_sequence_size, feature_length)

    rgb_input = Input(shape=input_shape, name='rgb_input')
    rgb_lstm = LSTM(2560, return_sequences=False, dropout=0.5, name='rgb_lstm')(rgb_input)
    rgb_dense1 = Dense(512, name='rgb_dense1')(rgb_lstm)
    rgb_dropout = Dropout(0.5, name='rgb_dropout')(rgb_dense1)
    rgb_dense_final = Dense(classes_size, activation='softmax', name='rgb_dense_final')(rgb_dropout)

    flow_input = Input(shape=input_shape, name='flow_input')
    flow_lstm = LSTM(2560, return_sequences=False, dropout=0.5, name='flow_lstm')(flow_input)
    flow_dense1 = Dense(512, name='flow_dense1')(flow_lstm)
    flow_dropout = Dropout(0.5, name='flow_dropout')(flow_dense1)
    flow_dense_final = Dense(classes_size, activation='softmax', name='flow_dense_final')(flow_dropout)

    model = Model(inputs=[rgb_input, flow_input], outputs=[rgb_dense_final, flow_dense_final])
    model.load_weights(os.path.join(project_root, 'data', 'result', 'model_weights', 'tsn_model-020-0.980-0.523.hdf5'))

    # Hyper parameters
    loss = 'categorical_crossentropy'
    optimizer = Adam(lr=1e-5, decay=1e-6)
    metrics = ['accuracy', 'top_k_categorical_accuracy']

    # Model created
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.summary()
    return model


def validate_model(model, validation_data):

    n_video = len(validation_data)
    correct_prediction = 0
    rgb_prediction = 0
    flow_prediction = 0

    rgb_classification = np.zeros((len(classes), len(classes)))
    flow_classification = np.zeros((len(classes), len(classes)))
    global_classification = np.zeros((len(classes), len(classes)))

    flow_folder_suffix = '_flowfeatures'
    flow_dataset_folder_name = 'UCF-101-flow-features'
    rgb_folder_suffix = '_features'
    rgb_dataset_folder_name = 'UCF-101-rgb-features'

    for video in validation_data:
        rgb_feature_filename_pattern = os.path.join(project_root, 'data',
                                                    rgb_dataset_folder_name, video.label,
                                                    video.label + rgb_folder_suffix,
                                                    'v_' + video.label + '_' + video.group + '_' + video.clip + '*.npy')

        flow_feature_filename_pattern = os.path.join(project_root, 'data',
                                                     flow_dataset_folder_name, video.label,
                                                     video.label + flow_folder_suffix,
                                                     'v_' + video.label + '_' + video.group + '_' + video.clip + '*.npy')

        rgb_segment_features_list = glob.glob(rgb_feature_filename_pattern)
        flow_segment_features_list = glob.glob(flow_feature_filename_pattern)

        if len(rgb_segment_features_list) != number_of_segment:
            raise Exception('A video has a wrong number of feature part. Please check your parameters')
        if len(flow_segment_features_list) != number_of_segment:
            raise Exception('A video has a wrong number of feature part. Please check your parameters')

        features_rgb = []
        features_flow = []
        for rgb_segment_feature in rgb_segment_features_list:

            if not os.path.isfile(rgb_segment_feature):
                raise Exception('Feature not found! You have to create all the features!')

            # Retrieving features
            features_sequence = load_and_pad(rgb_segment_feature, feature_sequence_size, feature_length)
            features_rgb.append(features_sequence)

        features_rgb = np.array(features_rgb).reshape((number_of_segment, feature_sequence_size, feature_length))

        for flow_segment_feature in flow_segment_features_list:

            if not os.path.isfile(flow_segment_feature):
                raise Exception('Feature not found! You have to create all the features!')

            # Retrieving features
            features_sequence = load_and_pad(flow_segment_feature, feature_sequence_size, feature_length)
            features_flow.append(features_sequence)

        features_flow = np.array(features_flow).reshape((number_of_segment, feature_sequence_size, feature_length))

        prediction = model.predict([features_rgb, features_flow], batch_size=number_of_segment, verbose=1)

        local_consensus, rgb_predict, flow_predict = compute_local_consensus(prediction)
        global_consensus = compute_global_consensus(local_consensus)

        if video.label == classes[global_consensus.argmax()]:
            correct_prediction += 1
        if video.label == classes[rgb_predict]:
            rgb_prediction += 1
        if video.label == classes[flow_predict]:
            flow_prediction += 1

        global_classification[classes.index(video.label)][global_consensus.argmax()] += 1
        rgb_classification[classes.index(video.label)][rgb_predict] += 1
        flow_classification[classes.index(video.label)][flow_predict] += 1

    accuracy = correct_prediction / n_video
    rgb_accuracy = rgb_prediction / n_video
    flow_accuracy = flow_prediction / n_video
    metrics = [accuracy, rgb_accuracy, flow_accuracy]
    save_result(metrics)
    return metrics


def compute_local_consensus(prediction):
    prediction = np.array(prediction)
    num_of_prediction = prediction.shape[0]
    prediction_length = prediction.shape[2]

    prediction = np.average(prediction, axis=1)
    max_indices = prediction.argmax(axis=1)

    prediction = np.zeros((num_of_prediction, prediction_length))
    for i in range(num_of_prediction):
        max_ind = max_indices[i]
        prediction[i][max_ind] = 1
    return prediction, max_indices[0], max_indices[1]


def compute_global_consensus(local_consensus):
    prediction = np.average(local_consensus, axis=0)
    max_ind = prediction.argmax()
    prediction = np.zeros(prediction.size)
    prediction[max_ind] = 1
    return prediction


def save_result(metrics):
    timestamp = time.time()
    filelog_name = 'exp3-validation-' + str(timestamp) + '.csv'
    with open(os.path.join(project_root, 'data', 'result', filelog_name), 'w', newline='\n') as val_result_csv:
        val_result_csv_writer = csv.writer(val_result_csv, delimiter=',')
        val_result_csv_writer.writerow(['val_accuracy', 'rgb_val_accuracy', 'flow_val_accuracy'])
        val_result_csv_writer.writerow(metrics)


if __name__ == '__main__':
    main()
