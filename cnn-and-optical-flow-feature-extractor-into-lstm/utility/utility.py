import csv
import os
import random

from utility.video_object import VideoObject


def project_root():
    return 'PATH TO cnn-and-optical-flow-feature-extractor-into-lstm'


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

    # Generate 'size_limit' number of random index
    new_frame_list_index = list(range(len(frame_list)))
    random.shuffle(new_frame_list_index)
    new_frame_list_index = new_frame_list_index[:size_limit]

    # Create new list
    new_frame_list = []
    for i in new_frame_list_index:
        new_frame_list.append(frame_list[i])
    new_frame_list.sort()
    return new_frame_list
