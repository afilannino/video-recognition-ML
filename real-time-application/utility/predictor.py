import os
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

from utility.utility import retrieve_classes, project_root

project_root = project_root()


class Predictor(object):

    def __init__(self, feature_sequence_size):
        self.feature_sequence_size = feature_sequence_size
        self.feature_length = 2048
        self.classes = retrieve_classes()
        self.model = self.create_model()

    def create_model(self):
        input_shape = (self.feature_sequence_size, self.feature_length)

        rgb_input = Input(shape=input_shape, name='rgb_input')
        rgb_lstm = LSTM(2560, return_sequences=False, dropout=0.5, name='rgb_lstm')(rgb_input)
        rgb_dense1 = Dense(512, name='rgb_dense1')(rgb_lstm)
        rgb_dropout = Dropout(0.5, name='rgb_dropout')(rgb_dense1)
        rgb_dense_final = Dense(len(self.classes), activation='softmax', name='rgb_dense_final')(rgb_dropout)

        flow_input = Input(shape=input_shape, name='flow_input')
        flow_lstm = LSTM(2560, return_sequences=False, dropout=0.5, name='flow_lstm')(flow_input)
        flow_dense1 = Dense(512, name='flow_dense1')(flow_lstm)
        flow_dropout = Dropout(0.5, name='flow_dropout')(flow_dense1)
        flow_dense_final = Dense(len(self.classes), activation='softmax', name='flow_dense_final')(flow_dropout)

        model = Model(inputs=[rgb_input, flow_input], outputs=[rgb_dense_final, flow_dense_final])

        # model.load_weights(
        #     os.path.join(project_root, 'data', 'result', 'model_weights', 'CHANGE_ME')
        # )
        model.summary()

        return model
