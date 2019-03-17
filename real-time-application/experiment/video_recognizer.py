import os
import numpy as np

from utility.frame_extractor import FrameExtractor
from utility.feature_extractor import FeatureExtractor
from utility.predictor import Predictor
from utility.utility import project_root, retrieve_classes

video_file_path = os.path.join(project_root(), 'data', 'video_test.avi')


feature_sequence_size = 10
number_of_segment = 3
classes = retrieve_classes()


def main():
    frame_extractor = FrameExtractor()
    feature_extractor = FeatureExtractor(feature_sequence_size)
    predictor = Predictor(feature_sequence_size)

    generator = frame_extractor.frames_generator(video_file_path)

    # inizializzo arrays di features, due array per ogni segmento e considero number_of_segments segmenti per volta
    # segment1 = []
    # segment2 = []
    # segment3 = []
    # current_features = np.ndarray([segment1, segment2, segment3])
    current_features = np.zeros((number_of_segment, 2, feature_sequence_size, 2048))
    class_recognized = np.zeros((len(classes)))

    index = 0

    while True:
        rgb_frames, flow_frames, additional_info = generator.__next__()
        if rgb_frames == 'STOP' or flow_frames == 'STOP':
            break

        rgb_features, flow_features = feature_extractor.extract_feature(rgb_frames, flow_frames)
        current_features[index] = [rgb_features, flow_features]
        index = increase_index(index, number_of_segment)

        current_prediction = predictor.model.predict(current_features, batch_size=number_of_segment)
        local_consensus, rgb_predict, flow_predict = compute_local_consensus(current_prediction, class_recognized)
        class_recognized = compute_global_consensus(local_consensus)

        if additional_info[0]:
            parallel = 1
        else:
            parallel = 2
        print('Segment n. %s on parallel flow n. %d is an action of '.format(additional_info[1], parallel)
              + classes[class_recognized.argmax()])

    print('End')


def compute_local_consensus(current_prediction, class_recognized):
    prediction = np.array(current_prediction)
    num_of_prediction = prediction.shape[0]
    prediction_length = prediction.shape[2]

    prediction = np.average(prediction, axis=1)  # change here for maximum

    # logic for including also the previous class_recognized
    for i in range(num_of_prediction):
        prediction[i] = prediction[i] * 0.75 + np.array(class_recognized) * 0.25

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


def increase_index(index, max_value):
    return (index + 1) % max_value


if __name__ == '__main__':
    main()
