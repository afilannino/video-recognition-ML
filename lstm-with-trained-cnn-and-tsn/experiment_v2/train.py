import os.path
import time
import math
import glob
import numpy as np
import csv

from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping, Callback, LambdaCallback
from keras.layers import Dense, Dropout, Input
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model

from utility.utility import retrieve_videoobject_subsets, project_root, retrieve_classes
from utility.experiment_utilities import create_sequence_generator_for_v2, load_and_pad

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
    epoch_number = 50

    subsets = retrieve_videoobject_subsets(['train', 'validation'])

    # Create final model and retrieve training data
    model = create_model()

    # Train model on train set
    model = train_model(model, subsets[0], subsets[1], batch_size, epoch_number)


def create_model():
    # input_shape = (feature_sequence_size, feature_length)
    #
    # rgb_input = Input(shape=input_shape, name='rgb_input')
    # rgb_lstm = LSTM(2560, return_sequences=False, dropout=0.5, name='rgb_lstm')(rgb_input)
    # rgb_dense1 = Dense(512, name='rgb_dense1')(rgb_lstm)
    # rgb_dropout = Dropout(0.5, name='rgb_dropout')(rgb_dense1)
    # rgb_dense_final = Dense(classes_size, activation='softmax', name='rgb_dense_final')(rgb_dropout)
    #
    # flow_input = Input(shape=input_shape, name='flow_input')
    # flow_lstm = LSTM(2560, return_sequences=False, dropout=0.5, name='flow_lstm')(flow_input)
    # flow_dense1 = Dense(512, name='flow_dense1')(flow_lstm)
    # flow_dropout = Dropout(0.5, name='flow_dropout')(flow_dense1)
    # flow_dense_final = Dense(classes_size, activation='softmax', name='flow_dense_final')(flow_dropout)

    input_shape = (feature_sequence_size * 2, feature_length)

    input = Input(shape=input_shape, name='input')
    lstm = LSTM(2560, return_sequences=False, dropout=0.5, name='lstm')(input)
    dense1 = Dense(512, name='dense1')(lstm)
    dropout = Dropout(0.5, name='dropout')(dense1)
    dense_final = Dense(classes_size, activation='softmax', name='dense_final')(dropout)

    model = Model(inputs=input, outputs=dense_final)
    # model.load_weights(os.path.join(project_root, 'data', 'result', 'model_weights', 'CHANGE_ME'))

    # Hyper parameters
    loss = 'categorical_crossentropy'
    optimizer = Adam(lr=1e-5, decay=1e-6)
    metrics = ['accuracy', 'top_k_categorical_accuracy']

    # plot_model(model, to_file='model.png', show_shapes=True)

    # Model created
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.summary()
    return model


def train_model(model, train_data, validation_data, batch_size, epoch_number):
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

    # Callback: EarlyStopping
    es = EarlyStopping(patience=8)

    train_steps_per_epoch = math.ceil(len(train_data) / batch_size)
    train_sequence_generator = create_sequence_generator_for_v2(
        train_data,
        classes,
        batch_size,
        feature_sequence_size,
        feature_length,
        number_of_segment
    )

    validation_steps_per_epoch = math.ceil(len(validation_data) / batch_size)
    validation_sequence_generator = create_sequence_generator_for_v2(
        validation_data,
        classes,
        batch_size,
        feature_sequence_size,
        feature_length,
        number_of_segment
    )

    model.fit_generator(generator=train_sequence_generator,
                        steps_per_epoch=train_steps_per_epoch,
                        epochs=epoch_number,
                        validation_data=validation_sequence_generator,
                        validation_steps=validation_steps_per_epoch,
                        verbose=1,
                        callbacks=[model_saver, csv_logger, es])

    # create_plot(filelog_name)
    print('Model trained!')
    return model


if __name__ == '__main__':
    main()
