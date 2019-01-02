import numpy as np
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.preprocessing import image


class InceptionV3Model:
    def __init__(self):
        # load model with inceptionV3 pre trained on imagenet
        inceptionv3_model = InceptionV3(weights='imagenet', include_top=True)

        self.model = Model(
            inputs=inceptionv3_model.input,
            outputs=inceptionv3_model.get_layer('avg_pool').output
        )

    def extract_features(self, frame_path):
        frame = image.load_img(frame_path, target_size=(299, 299))
        x = image.img_to_array(frame)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Retrieve features by means of prediction
        features = self.model.predict(x)
        features = features[0]
        return features
