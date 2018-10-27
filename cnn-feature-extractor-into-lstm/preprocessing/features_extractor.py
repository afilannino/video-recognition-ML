import glob
import os
import random

import numpy as np
import tqdm

from .frames_generator import retrieve_videoobject_subsets
from .inceptionV3 import InceptionV3Model

project_root = 'INSERIRE PATH ROOT PROGETTO'


def main():
    videoobject_subsets = retrieve_videoobject_subsets(['train', 'validation', 'test'])
    model = InceptionV3Model()
    extract_features(videoobject_subsets, model, size_limit=100)


def extract_features(videoobject_subsets, model, size_limit=100, skip_existent=True):
    # Initializing progress bar
    length = 0
    for videoobject_subset in videoobject_subsets:
        length += len(videoobject_subset)
    print('Starting feature extraction from each frames in subsets')
    progress_bar = tqdm.tqdm(total=length)

    # Starting looping over subset of videos
    for videoobject_subset in videoobject_subsets:

        for video in videoobject_subset:
            video_base = os.path.join(project_root, 'data', 'UCF-101', video.label,
                                      'v_' + video.label + '_' + video.group + '_' + video.clip)
            frame_folder_name = video_base + '_frames'
            features_folder_name = os.path.join(project_root, 'data', 'UCF-101', video.label, video.label + '_features')
            feature_filename = os.path.join(features_folder_name,
                                            'v_' + video.label + '_' + video.group + '_' + video.clip)
            if not os.path.exists(features_folder_name):
                os.mkdir(features_folder_name)

            # Skip folder if features have been already extracted
            if skip_existent:
                if os.path.exists(feature_filename + '.npy'):
                    progress_bar.update(1)
                    continue

            # Retrieve frames for this video
            frames = glob.glob(os.path.join(frame_folder_name, 'frame-*.jpg'))
            frames = limit_frames_size(frames, size_limit)

            # Loop over all frames, extract features and concatenate in a list
            features_sequence = []
            for frame in frames:
                features = model.extract_features(frame)
                features_sequence.append(features)

            # Save sequence of feature as a numpy array
            np.save(feature_filename, features_sequence)

            progress_bar.update(1)

    progress_bar.close()


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


if __name__ == '__main__':
    main()
