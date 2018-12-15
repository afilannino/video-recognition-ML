import os.path
import time
import math

from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping
from keras.layers import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.optimizers import Adam

from utility.utility import retrieve_videoobject_subsets, project_root, retrieve_classes
from utility.experiment_utilities import create_sequence_generator, retrieve_sequence
from experiment.plot_data import main as create_plot

project_root = project_root()
feature_length = 2048  # This is the length of the numpy array that contains the prediction
classes_size = len(retrieve_classes())  # Number of classes in UCF101 dataset
feature_sequence_size = 30  # This must be coherent with the value used in features_extractor.py


def main():
    preload_features_in_memory = False

    # Hyper parameters
    batch_size = 64
    epoch_number = 200

    # Create final model and retrieve training data
    model = create_final_model()
    classes = retrieve_classes()
    subsets = retrieve_videoobject_subsets(['train', 'validation'])

    # Train model on train set
    train(model, classes, subsets[0], subsets[1],
          preload_features_in_memory=preload_features_in_memory,
          batch_size=batch_size,
          epoch_number=epoch_number)


def create_final_model(model_weights=False, weights_path=None):
    input_shape = (feature_sequence_size, feature_length)

    model = Sequential()
    model.add(LSTM(1536, return_sequences=False, input_shape=input_shape, dropout=0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes_size, activation='softmax'))

    # Hyper parameters
    loss = 'categorical_crossentropy'
    optimizer = Adam(lr=1e-5, decay=1e-6)
    metrics = ['accuracy', 'top_k_categorical_accuracy']

    if model_weights and weights_path is not None:
        model.load_weights(weights_path)

    # Model created
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.summary()
    return model


def train(model, classes, train_data, validation_data, preload_features_in_memory=False, batch_size=128,
          epoch_number=100):
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

    if preload_features_in_memory:
        # Reading and reshaping training data
        x_train, y_train = retrieve_sequence(train_data, classes, feature_sequence_size, feature_length)
        # x_train = np.reshape(x_train, (x_train.shape[0], feature_sequence_size, feature_length))
        # y_train = np.reshape(y_train, (y_train.shape[0], len(classes))

        # Reading and reshaping validation data
        x_validation, y_validation = retrieve_sequence(validation_data, classes, feature_sequence_size, feature_length)
        # x_validation = np.reshape(x_validation, (x_validation.shape[0], feature_sequence_size, feature_length))
        # y_validation = np.reshape(y_validation, (y_validation.shape[0], len(classes))

        # Training
        model.fit(x_train, y_train,
                  validation_data=(x_validation, y_validation),
                  batch_size=batch_size,
                  epochs=epoch_number,
                  verbose=1,
                  callbacks=[model_saver, csv_logger, tb, es])
    else:
        train_steps_per_epoch = math.ceil(len(train_data) / batch_size)
        validation_steps_per_epoch = math.ceil(len(validation_data) / batch_size)
        train_sequence_generator = create_sequence_generator(train_data, classes, batch_size, feature_sequence_size,
                                                             feature_length)
        validation_sequence_generator = create_sequence_generator(validation_data, classes, batch_size,
                                                                  feature_sequence_size, feature_length)
        model.fit_generator(generator=train_sequence_generator,
                            steps_per_epoch=train_steps_per_epoch,
                            validation_data=validation_sequence_generator,
                            validation_steps=validation_steps_per_epoch,
                            epochs=epoch_number,
                            verbose=1,
                            callbacks=[model_saver, csv_logger, tb, es])

    create_plot(filelog_name)
    print('Model trained!')


if __name__ == '__main__':
    main()
