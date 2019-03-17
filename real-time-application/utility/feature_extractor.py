import os
import numpy as np

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image

from utility.utility import project_root, retrieve_classes, zero_pad_sequence

project_root = project_root()
classes = retrieve_classes()


class FeatureExtractor(object):

    def __init__(self, feature_sequence_size):
        self.inception_v3 = self.create_trained_inceptionv3_model()
        self.feature_length = 2048
        self.feature_sequence_size = feature_sequence_size

    def extract_feature(self, rgb_frames, flow_frames):
        rgb_features_sequence = []
        flow_features_sequence = []

        for frame in rgb_frames:
            x = image.img_to_array(frame)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            features = self.inception_v3.predict(x)
            rgb_features_sequence.append(features[0])

        if flow_frames != 0:

            for frame in flow_frames:
                x = image.img_to_array(frame)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                features = self.inception_v3.predict(x)
                flow_features_sequence.append(features[0])

        return (zero_pad_sequence(rgb_features_sequence, self.feature_sequence_size, self.feature_length),
                zero_pad_sequence(flow_features_sequence, self.feature_sequence_size, self.feature_length))

    @staticmethod
    def create_trained_inceptionv3_model():
        # model_weights = os.path.join(project_root, 'data', 'result', 'model_weights', 'cnn-training-CHANGEME')

        # if not os.path.exists(model_weights):
        #    raise Exception('No inception v3 model weights found!')

        base_model = InceptionV3(weights='imagenet', include_top=False)
        x = base_model.output
        x = GlobalAveragePooling2D(name='final_avg_pool')(x)
        # x = Dense(len(classes), activation='softmax', name='final_dense')(x)

        model = Model(
            inputs=base_model.input,
            outputs=x
        )
        # model.load_weights(model_weights)
        return model
