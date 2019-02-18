import os.path
import time
import math

from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping
from keras.layers import Dense, Dropout, Input, Average, average
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model

from utility.utility import retrieve_videoobject_subsets, project_root, retrieve_classes
from utility.experiment_utilities import create_sequence_generator, create_sequence_of_tuple_generator
from experiment.plot_data import main as create_plot

project_root = project_root()
feature_length = 2048  # This is the length of the numpy array that contains the prediction
classes_size = len(retrieve_classes())  # Number of classes in UCF101 dataset
feature_sequence_size = 10  # This must be coherent with the value used in features_extractor.py
final_layer_ouput_length = 256
number_of_segment = 3


def main():
    # Hyper parameters
    batch_size = 66
    epoch_number = 200

    # Create final model and retrieve training data
    model_rgb, model_flow = create_partial_models()

    classes = retrieve_classes()
    subsets = retrieve_videoobject_subsets(['train', 'validation'])

    # Train rgb model on train set
    model_rgb = train_partial_model(model_rgb, classes, subsets[0], subsets[1], 'rgb',
                                    batch_size=batch_size,
                                    epoch_number=epoch_number)

    # Train flow model on train set
    model_flow = train_partial_model(model_flow, classes, subsets[0], subsets[1], 'flow',
                                     batch_size=batch_size,
                                     epoch_number=epoch_number)

    final_model = create_final_model()
    train_final_model(final_model, model_rgb, model_flow, classes, subsets[0], subsets[1],
                      batch_size=batch_size,
                      epoch_number=epoch_number)


def create_partial_models():

    input_shape = (feature_sequence_size, feature_length)

    rgb_input = Input(shape=input_shape, name='rgb_input')
    rgb_lstm = LSTM(2048, return_sequences=False, dropout=0.5, name='rgb_lstm')(rgb_input)
    rgb_dense1 = Dense(512, name='rgb_dense1')(rgb_lstm)
    rgb_dropout = Dropout(0.5, name='rgb_dropout')(rgb_dense1)
    rgb_dense2 = Dense(final_layer_ouput_length, name='rgb_dense2')(rgb_dropout)
    rgb_dense_final = Dense(classes_size, activation='softmax', name='rgb_dense_final')(rgb_dense2)

    flow_input = Input(shape=input_shape, name='flow_input')
    flow_lstm = LSTM(2048, return_sequences=False, dropout=0.5, name='flow_lstm')(flow_input)
    flow_dense1 = Dense(512, name='flow_dense1')(flow_lstm)
    flow_dropout = Dropout(0.5, name='flow_dropout')(flow_dense1)
    flow_dense2 = Dense(final_layer_ouput_length, name='flow_dense2')(flow_dropout)
    flow_dense_final = Dense(classes_size, activation='softmax', name='flow_dense_final')(flow_dense2)

    model_rgb = Model(inputs=rgb_input, outputs=rgb_dense_final)
    model_flow = Model(inputs=flow_input, outputs=flow_dense_final)

    # Hyper parameters
    loss = 'categorical_crossentropy'
    optimizer = Adam(lr=1e-5, decay=1e-6)
    metrics = ['accuracy', 'top_k_categorical_accuracy']

    plot_model(model_rgb, to_file='model_partial_rgb.png', show_shapes=True)
    plot_model(model_flow, to_file='model_partial_flow.png', show_shapes=True)

    # Model created
    model_rgb.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model_rgb.summary()
    model_flow.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model_flow.summary()
    return model_rgb, model_flow


def train_partial_model(model, classes, train_data, validation_data, type, batch_size=128, epoch_number=100):
    timestamp = time.time()
    # Callback: function to save the model weights
    model_saver = ModelCheckpoint(
        filepath=os.path.join(project_root, 'data', 'result', 'model_weights',
                              'partial_model_'+type+'-{epoch:03d}-{val_loss:.3f}.hdf5'),
        verbose=1,
        save_best_only=True)

    # Callback: function to log result
    filelog_name = type + '_partial_model-training-' + str(timestamp) + '.log'
    log_path = os.path.join(project_root, 'data', 'result', 'logs', filelog_name)
    csv_logger = CSVLogger(log_path)

    # Callback: TensorBoard
    tb = TensorBoard(log_dir=os.path.join(project_root, 'data', 'result', 'logs'))

    # Callback: EarlyStopping
    es = EarlyStopping(patience=5)

    train_steps_per_epoch = math.ceil(len(train_data) / batch_size)
    validation_steps_per_epoch = math.ceil(len(validation_data) / batch_size)

    train_sequence_generator = create_sequence_generator(
        train_data, classes, batch_size, feature_sequence_size, feature_length, type, number_of_segment)
    validation_sequence_generator = create_sequence_generator(
        validation_data, classes, batch_size, feature_sequence_size, feature_length, type, number_of_segment)

    model.fit_generator(generator=train_sequence_generator,
                        validation_data=validation_sequence_generator,
                        steps_per_epoch=train_steps_per_epoch,
                        validation_steps=validation_steps_per_epoch,
                        epochs=epoch_number,
                        verbose=1,
                        callbacks=[model_saver, csv_logger, es])

    create_plot(filelog_name)
    print(type + ' partial model trained!')
    return model


def create_final_model():

    input_shape = (number_of_segment, final_layer_ouput_length)

    consensus_dense_output = 256

    rgb_consensus_input = Input(shape=input_shape, name='rgb_consensus_input')
    rgb_consensus_avg = average(rgb_consensus_input, name='rgb_consensus_avg')
    rgb_consensus_dense = Dense(consensus_dense_output, name='rgb_consensus_dense')(rgb_consensus_avg)

    flow_consensus_input = Input(shape=input_shape, name='flow_consensus_input')
    flow_consensus_avg = average(flow_consensus_input, name='flow_consensus_avg')
    flow_consensus_dense = Dense(consensus_dense_output, name='flow_consensus_dense')(flow_consensus_avg)

    final_avg = average([rgb_consensus_dense, flow_consensus_dense])
    final_dense = Dense(classes_size, activation='softmax', name='final_dense')(final_avg)

    model = Model(inputs=[rgb_consensus_input, flow_consensus_input], outputs=final_dense)

    # Hyper parameters
    loss = 'categorical_crossentropy'
    optimizer = Adam(lr=1e-5, decay=1e-6)
    metrics = ['accuracy', 'top_k_categorical_accuracy']

    # Model created
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.summary()
    return model


def train_final_model(model, model_rgb, model_flow, classes, train_data, validation_data,
                      batch_size=200,
                      epoch_number=200):
    timestamp = time.time()
    # Callback: function to save the model weights
    model_saver = ModelCheckpoint(
        filepath=os.path.join(project_root, 'data', 'result', 'model_weights',
                              'finalmodel-{epoch:03d}-{val_loss:.3f}.hdf5'),
        verbose=1,
        save_best_only=True)

    # Callback: function to log result
    filelog_name = 'finalmodel-training-' + str(timestamp) + '.log'
    log_path = os.path.join(project_root, 'data', 'result', 'logs', filelog_name)
    csv_logger = CSVLogger(log_path)

    # Callback: TensorBoard
    tb = TensorBoard(log_dir=os.path.join(project_root, 'data', 'result', 'logs'))

    # Callback: EarlyStopping
    es = EarlyStopping(patience=5)

    train_steps_per_epoch = math.ceil(len(train_data) / batch_size)
    validation_steps_per_epoch = math.ceil(len(validation_data) / batch_size)

    train_sequence_generator = create_sequence_of_tuple_generator(
        train_data, batch_size, feature_sequence_size, feature_length, model_flow, model_rgb, number_of_segment)
    validation_sequence_generator = create_sequence_of_tuple_generator(
        validation_data, batch_size, feature_sequence_size, feature_length, model_flow, model_rgb, number_of_segment)

    model.fit_generator(generator=train_sequence_generator,
                        steps_per_epoch=train_steps_per_epoch,
                        validation_data=validation_sequence_generator,
                        validation_steps=validation_steps_per_epoch,
                        epochs=epoch_number,
                        verbose=1,
                        callbacks=[model_saver, csv_logger, es])

    create_plot(filelog_name)
    print('Model trained!')


if __name__ == '__main__':
    main()
