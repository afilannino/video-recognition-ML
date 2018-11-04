import csv
import os
import numpy as np

from utility.video_object import VideoObject


def project_root():
    return 'PATH TO cnn-feature-extractor-into-lstm'


def ffmpeg_path():
    return 'PATH TO ffmpeg'


def retrieve_videoobject_subsets(subsets):
    videoobject_subsets = []
    # Loop over the three subsets: train, validation, test
    for subset in subsets:
        videoobject_subset = []
        # Open the csv file and retrieve a list of VideoObject
        with open(os.path.join(project_root(), 'data', subset + '-set.csv'), 'r', newline='\n') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            next(csv_reader)  # Skip the header
            for row in csv_reader:
                videoobject_subset.append(VideoObject(row[0], row[1], row[2]))
        videoobject_subsets.append(videoobject_subset)
    return videoobject_subsets


def limit_frames_size(frame_list, size_limit):
    # Check if requirement are already satisfied
    if len(frame_list) <= size_limit:
        return frame_list

    # Pick 'size_limit' frames from the frame list and reorder them
    new_frame_list = np.random.choice(frame_list, size=size_limit, replace=False).tolist()
    new_frame_list.sort()
    return new_frame_list


def retrieve_classes():
    classes = []
    with open(os.path.join(project_root(), 'data', 'classes.csv'), 'r', newline='\n') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            classes.append(row[0])
    return classes
