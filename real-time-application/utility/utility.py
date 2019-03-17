import os
import csv
import numpy as np


def project_root():
    return 'D:\\GIT_repository\\video-recognition-ML\\real-time-application'
    # return 'PATH TO real-time-application folder'


def limit_frames_number(frames_list, limit):
    # Check if requirement are already satisfied
    if len(frames_list) <= limit:
        return frames_list

    step = len(frames_list) / limit
    index = map(lambda x: round(x * step), list(range(limit)))

    new_frame_list = [frames_list[i] for i in index]
    # new_frame_list.sort()  # TODO check
    return new_frame_list[:limit]


def retrieve_classes():
    classes = []
    with open(os.path.join(project_root(), 'data', 'classes.csv'), 'r', newline='\n') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            classes.append(row[0])
    return classes


def zero_pad_sequence(features_sequence, feature_sequence_length, feature_length):
    length = len(features_sequence)
    if length > feature_sequence_length:
        raise Exception('Some sequences of features are too long! Please compute them again!')
    # Zero padding if size is different from size_limit
    elif length < feature_sequence_length:
        features_sequence = np.append(features_sequence,
                                      np.zeros((feature_sequence_length - length, feature_length),
                                               dtype='float32'))
        features_sequence = features_sequence.reshape((feature_sequence_length, feature_length))

    return features_sequence
