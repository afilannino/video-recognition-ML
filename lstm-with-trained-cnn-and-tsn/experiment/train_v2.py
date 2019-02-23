import os.path
import time
import math
import glob
import numpy as np
import csv

from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping
from keras.layers import Dense, Dropout, Input
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model

from utility.utility import retrieve_videoobject_subsets, project_root, retrieve_classes
from utility.experiment_utilities import create_sequence_generator

project_root = project_root()
feature_length = 2048  # This is the length of the numpy array that contains the prediction
classes_size = len(retrieve_classes())  # Number of classes in UCF101 dataset
feature_sequence_size = 10  # This must be coherent with the value used in features_extractor.py
final_layer_ouput_length = 256
number_of_segment = 3
classes = retrieve_classes()


def main():
    # Hyper parameters
    batch_size = 66
    epoch_number = 40

    subsets = retrieve_videoobject_subsets(['train', 'validation'])

    # Create final model and retrieve training data
    model = create_model()

    # Train model on train set
    model = train_model(model, subsets[0], batch_size, epoch_number)

    # Validate model
    metrics = validate_model(model, subsets[1])
    save_result(metrics)


def create_model():
    input_shape = (feature_sequence_size, feature_length)

    rgb_input = Input(shape=input_shape, name='rgb_input')
    rgb_lstm = LSTM(2048, return_sequences=False, dropout=0.5, name='rgb_lstm')(rgb_input)
    rgb_dense1 = Dense(512, name='rgb_dense1')(rgb_lstm)
    rgb_dropout = Dropout(0.5, name='rgb_dropout')(rgb_dense1)
    rgb_dense_final = Dense(classes_size, activation='softmax', name='rgb_dense_final')(rgb_dropout)

    flow_input = Input(shape=input_shape, name='flow_input')
    flow_lstm = LSTM(2048, return_sequences=False, dropout=0.5, name='flow_lstm')(flow_input)
    flow_dense1 = Dense(512, name='flow_dense1')(flow_lstm)
    flow_dropout = Dropout(0.5, name='flow_dropout')(flow_dense1)
    flow_dense_final = Dense(classes_size, activation='softmax', name='flow_dense_final')(flow_dropout)

    model = Model(inputs=[rgb_input, flow_input], outputs=[rgb_dense_final, flow_dense_final])

    # Hyper parameters
    loss = 'categorical_crossentropy'
    optimizer = Adam(lr=1e-5, decay=1e-6)
    metrics = ['accuracy', 'top_k_categorical_accuracy']

    plot_model(model, to_file='model_partial_rgb.png', show_shapes=True)

    # Model created
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.summary()
    return model


def train_model(model, train_data, batch_size, epoch_number):
    timestamp = time.time()
    # Callback: function to save the model weights
    model_saver = ModelCheckpoint(
        filepath=os.path.join(project_root, 'data', 'result', 'model_weights',
                              'tsn_model-{epoch:03d}-{val_loss:.3f}.hdf5'),
        verbose=1,
        save_best_only=True)

    # Callback: function to log result
    filelog_name = 'tsn_model-training-' + str(timestamp) + '.log'
    log_path = os.path.join(project_root, 'data', 'result', 'logs', filelog_name)
    csv_logger = CSVLogger(log_path)

    # Callback: TensorBoard
    tb = TensorBoard(log_dir=os.path.join(project_root, 'data', 'result', 'logs'))

    # Callback: EarlyStopping
    es = EarlyStopping(patience=5)

    train_steps_per_epoch = math.ceil(len(train_data) / batch_size)

    train_sequence_generator = create_sequence_generator(
        train_data, classes, batch_size, feature_sequence_size, feature_length, type, number_of_segment)

    model.fit_generator(generator=train_sequence_generator,
                        steps_per_epoch=train_steps_per_epoch,
                        epochs=epoch_number,
                        verbose=1,
                        callbacks=[model_saver, csv_logger, es])

    # create_plot(filelog_name)
    print('Model trained!')
    return model


def validate_model(model, validation_data):

    n_video = len(validation_data)
    correct_prediction = 0

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
            features_sequence = np.load(rgb_segment_feature)
            features_rgb.append(features_sequence)

        features_rgb = np.array(features_rgb).reshape((number_of_segment, feature_sequence_size, feature_length))

        for flow_segment_feature in flow_segment_features_list:

            if not os.path.isfile(flow_segment_feature):
                raise Exception('Feature not found! You have to create all the features!')

            # Retrieving features
            features_sequence = np.load(flow_segment_feature)
            features_flow.append(features_sequence)

        features_flow = np.array(features_flow).reshape((number_of_segment, feature_sequence_size, feature_length))

        prediction = model.predict([features_rgb, features_flow], batch_size=number_of_segment, verbose=1)

        local_consensus = compute_local_consensus(prediction)
        global_consensus = compute_global_consensus(local_consensus)

        if video.label == classes[global_consensus.argmax()]:
            correct_prediction += 1

    accuracy = correct_prediction / n_video
    return [accuracy]


def compute_local_consensus(prediction):
    num_of_prediction = prediction.shape[0]
    prediction_length = prediction.shape[1]

    prediction = np.array(prediction)
    prediction = prediction.sum(axis=1)
    max_indices = prediction.argmax(axis=1)

    prediction = np.zeros((num_of_prediction, prediction_length))
    for i in range(num_of_prediction):
        max_ind = max_indices[i]
        prediction[i][max_ind] = 1
    return prediction


def compute_global_consensus(local_consensus):
    prediction = local_consensus.sum(axis=0)
    max_ind = prediction.argmax()
    prediction = np.zeros(prediction.size)
    prediction[max_ind] = 1
    return prediction


def save_result(metrics):
    timestamp = time.time()
    filelog_name = 'exp3-validation-' + str(timestamp) + '.csv'
    with open(os.path.join(project_root, 'data', 'result', filelog_name), 'w', newline='\n') as val_result_csv:
        val_result_csv_writer = csv.writer(val_result_csv, delimiter=',')
        val_result_csv_writer.writerow(['accuracy'])
        val_result_csv_writer.writerow(metrics)


if __name__ == '__main__':
    main()
