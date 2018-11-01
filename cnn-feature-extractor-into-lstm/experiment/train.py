import os.path
import time
import numpy as np

from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.layers import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.optimizers import Adam

from utility.utility import retrieve_videoobject_subsets, project_root, retrieve_classes
from utility.experiment_utilities import create_sequence_generator, retrieve_sequence

project_root = project_root()
feature_length = 2048  # This is the length of the numpy array that contains the prediction
classes_size = 101  # Number of classes in UCF101 dataset
feature_sequence_size = 100  # This must be coherent with the value used in features_extractor.py


def main():
    preload_features_in_memory = True

    # Hyper parameters
    batch_size = 32
    epoch_number = 1

    # Create final model and retrieve training data
    model = create_final_model()
    classes = retrieve_classes()
    subsets = retrieve_videoobject_subsets(['train', 'validation'])

    # Train model on train set
    train(model, classes, subsets[0], subsets[1], preload_features_in_memory=preload_features_in_memory,
          batch_size=batch_size, epoch_number=epoch_number)


def create_final_model():
    input_shape = (feature_sequence_size, feature_length)

    model = Sequential()
    model.add(LSTM(2048, return_sequences=False, input_shape=input_shape, dropout=0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes_size, activation='softmax'))

    # Hyper parameters
    loss = 'categorical_crossentropy'
    optimizer = Adam(lr=1e-5, decay=1e-6)
    metrics = ['accuracy', 'top_k_categorical_accuracy']

    # Model created
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    print(model.summary())
    return model


def train(model, classes, train_data, validation_data, preload_features_in_memory=False, batch_size=128,
          epoch_number=100):
    # Utility function to save the model weights
    model_saver = ModelCheckpoint(
        filepath=os.path.join(project_root, 'data', 'result', 'model_weights',
                              'finalmodel-{epoch:03d}-{val_loss:.3f}.hdf5'),
        verbose=1,
        save_best_only=True)

    # Utility function to log result
    timestamp = time.time()
    csv_logger = CSVLogger(os.path.join(project_root, 'data', 'result', 'logs',
                                        'finalmodel-training-' + str(timestamp) + '.log'))

    train_steps_per_epoch = int(len(train_data) // batch_size)
    validation_steps_per_epoch = int(len(validation_data) // batch_size)

    if preload_features_in_memory:
        # Reading and reshaping training data
        x_train, y_train = retrieve_sequence(train_data, classes)
        # x_train.reshape((x_train.shape[0], feature_sequence_size, feature_length))
        # y_train.reshape(y_train.shape[0], len(classes))

        # Reading and reshaping validation data
        x_validation, y_validation = retrieve_sequence(validation_data, classes)
        # x_validation.reshape(x_validation.shape[0], feature_sequence_size, feature_length)
        # y_validation.reshape(y_validation.shape[0], len(classes))

        # Training
        model.fit(x_train, y_train,
                  validation_data=(x_validation, y_validation),
                  batch_size=batch_size,
                  epochs=epoch_number,
                  verbose=1,
                  callbacks=[model_saver, csv_logger])
    else:
        train_sequence_generator = create_sequence_generator(train_data, classes, batch_size)
        validation_sequence_generator = create_sequence_generator(validation_data, classes, batch_size)
        model.fit_generator(generator=train_sequence_generator,
                            steps_per_epoch=train_steps_per_epoch,
                            validation_data=validation_sequence_generator,
                            validation_steps=validation_steps_per_epoch,
                            epochs=epoch_number,
                            verbose=1,
                            callbacks=[model_saver, csv_logger])
    print('Model trained!')


if __name__ == '__main__':
    main()
