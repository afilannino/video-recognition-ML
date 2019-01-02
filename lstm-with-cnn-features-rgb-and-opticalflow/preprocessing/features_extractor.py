import glob
import os
import numpy as np
import tqdm

from utility.utility import retrieve_videoobject_subsets, limit_frames_number, project_root
from preprocessing.inceptionV3 import InceptionV3Model

project_root = project_root()


def main():
    videoobject_subsets = retrieve_videoobject_subsets(['validation', 'train'])
    model = InceptionV3Model()
    extract_features(videoobject_subsets, model, size_limit=30, flow_feature=False)
    extract_features(videoobject_subsets, model, size_limit=30, flow_feature=True)


def extract_features(videoobject_subsets, model, size_limit=30, skip_existent=True, flow_feature=False):
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
            if flow_feature:
                frame_folder_name = video_base + '_flowframes'
                features_folder_name = os.path.join(project_root, 'data', 'UCF-101', video.label,
                                                    video.label + '_features')
            else:
                frame_folder_name = video_base + '_frames'
                features_folder_name = os.path.join(project_root, 'data', 'UCF-101', video.label,
                                                    video.label + '_flowfeatures')

            feature_filename = os.path.join(features_folder_name,
                                            'v_' + video.label + '_' + video.group + '_' + video.clip)
            if not os.path.exists(features_folder_name):
                os.mkdir(features_folder_name)

            # Skip folder if features have been already extracted
            if skip_existent:
                if os.path.exists(feature_filename + '.npy'):
                    progress_bar.update(1)
                    continue

            # Retrieve frames (or flowframes) for this video
            if flow_feature:
                frames = glob.glob(os.path.join(frame_folder_name, 'flowframe-*.jpg'))
            else:
                frames = glob.glob(os.path.join(frame_folder_name, 'frame-*.jpg'))

            frames = limit_frames_number(frames, size_limit)

            # Loop over all frames, extract features and concatenate in a list
            features_sequence = []
            for frame in frames:
                features = model.extract_features(frame)
                features_sequence.append(features)

            # Save sequence of feature as a numpy array
            np.save(feature_filename, features_sequence)

            progress_bar.update(1)

    progress_bar.close()


if __name__ == '__main__':
    main()
