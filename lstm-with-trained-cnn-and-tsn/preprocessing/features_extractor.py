import glob
import os

import numpy as np
import tqdm
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image

from utility.utility import retrieve_classes, project_root
from utility.utility import retrieve_videoobject_subsets, limit_frames_number

project_root = project_root()
features_lenght = 10
number_of_segment = 3
classes = retrieve_classes()


def main():
    videoobject_subsets = retrieve_videoobject_subsets(['train', 'validation'])
    model = create_trained_inceptionv3_model()
    # generate_and_store_features(videoobject_subsets, model, flow_feature=False)
    generate_and_store_features(videoobject_subsets, model, flow_feature=True)


def create_trained_inceptionv3_model():
    # model_weights = os.path.join(project_root, 'data', 'result_trainCNN_final', 'model_weights',
    #                             'cnn-training-010-6.011.hdf5')

    base_model = InceptionV3(weights='imagenet', include_top=True)
    # x = base_model.output
    # x = GlobalAveragePooling2D(name='final_avg_pool')(x)
    # x = Dense(1024, activation='relu', name='final_dense')(x)

    # predictions = Dense(len(classes), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
    return model
    # if os.path.exists(model_weights):
    #    model.load_weights(model_weights)
    # else:
    #   raise Exception('No model weights found!')

    # trained_model = Model(input=model.input, output=model.get_layer('final_avg_pool').output)
    # return trained_model


def generate_and_store_features(videoobject_subsets, model, skip_existent=True, flow_feature=False):
    # Initializing progress bar
    length = 0
    for videoobject_subset in videoobject_subsets:
        length += len(videoobject_subset)
    print('Starting feature extraction from each frames in subsets')
    progress_bar = tqdm.tqdm(total=length*number_of_segment)

    # Starting looping over subset of videos
    for videoobject_subset in videoobject_subsets:

        for video in videoobject_subset:
            video_base = os.path.join(project_root, 'data', 'UCF-101', video.label,
                                      'v_' + video.label + '_' + video.group + '_' + video.clip)
            if flow_feature:
                frame_folder_name = video_base + '_flowframes'
                features_folder_name = os.path.join(project_root, 'data', 'UCF-101-flow-features', video.label,
                                                    video.label + '_flowfeatures')
            else:
                frame_folder_name = video_base + '_frames'
                features_folder_name = os.path.join(project_root, 'data', 'UCF-101-rgb-features', video.label,
                                                    video.label + '_features')

            feature_filename = os.path.join(features_folder_name,
                                            'v_' + video.label + '_' + video.group + '_' + video.clip)
            if not os.path.exists(features_folder_name):
                os.makedirs(features_folder_name)

            # Skip folder if features have been already extracted
            if skip_existent:
                if len(glob.glob(os.path.join(feature_filename + '*.npy'))) is number_of_segment:
                    progress_bar.update(number_of_segment)
                    continue

            # Retrieve frames (or flowframes) for this video
            try:
                frames = os.listdir(frame_folder_name)
            except IOError:
                raise Exception('You have to generate frames and flowframes first!')

            segments = np.array_split(frames, number_of_segment)

            for index, segment in enumerate(segments):
                segment = limit_frames_number(segment, features_lenght)
                features_sequence = []
                for frame in segment:
                    features = extract_features(model, os.path.join(frame_folder_name, frame))
                    features_sequence.append(features)

                # Save sequence of feature as a numpy array
                np.save('{}_part{}'.format(feature_filename, index), features_sequence)
                progress_bar.update(1)

    progress_bar.close()


def extract_features(model, frame_path):
    frame = image.load_img(frame_path, target_size=(299, 299))
    x = image.img_to_array(frame)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Retrieve features by means of prediction
    features = model.predict(x)
    features = features[0]
    return features


if __name__ == '__main__':
    main()
