import os
import subprocess
import shutil
import glob

import tqdm
from utility.utility import retrieve_videoobject_subsets, project_root, ffmpeg_path

project_root = project_root()
ffmpeg_path = ffmpeg_path()


def main():
    # generate_frames(['train', 'validation'])
    generate_frames_for_cnn(['train', 'validation'])


def generate_frames(subsets, skip_existent=True):
    # Retrieve video's subsets
    videoobject_subsets = retrieve_videoobject_subsets(subsets)

    # Initializing progress bar
    length = 0
    for videoobject_subset in videoobject_subsets:
        length += len(videoobject_subset)
    print('Starting frames generation')
    progress_bar = tqdm.tqdm(total=length)

    # Starting looping over subset of videos
    for videoobject_subset in videoobject_subsets:

        for video in videoobject_subset:
            video_base = os.path.join(project_root, 'data', 'UCF-101', video.label,
                                      'v_' + video.label + '_' + video.group + '_' + video.clip)
            frame_folder_name = video_base + '_frames'
            video_filename = video_base + '.avi'
            if not os.path.exists(frame_folder_name):
                os.mkdir(frame_folder_name)

            # Skip folder if frames have been already generated
            if skip_existent:
                first_frame_filename = os.path.join(frame_folder_name, 'frame-0001.jpg')
                if os.path.exists(first_frame_filename):
                    progress_bar.update(1)
                    continue

            # Generate frames
            frames_name_pattern = os.path.join(frame_folder_name, 'frame-%04d.jpg')
            subprocess.call([ffmpeg_path, "-i", video_filename, frames_name_pattern],
                            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            progress_bar.update(1)
    progress_bar.close()


def generate_frames_for_cnn(subsets):
    # Retrieve video's subsets
    videoobject_subsets = retrieve_videoobject_subsets(subsets)

    # Initializing progress bar
    length = 0
    for videoobject_subset in videoobject_subsets:
        length += len(videoobject_subset)
    print('Starting frames generation')
    progress_bar = tqdm.tqdm(total=length)

    # Starting looping over subset of videos
    for videoobject_subset in videoobject_subsets:
        for video in videoobject_subset:
            #print(video.label)
            frame_folder_name = os.path.join(project_root, '..', 'lstm-with-cnn-features-rgb-and-opticalflow', 'data', 'UCF-101', video.label,
                                      'v_' + video.label + '_' + video.group + '_' + video.clip) + '_frames'
            files_to_copy = os.listdir(frame_folder_name)

            dest = os.path.join(project_root, 'data', 'UCF-101-cnn-frames', video.label)
            if not os.path.exists(dest):
                os.makedirs(dest)

            for f in files_to_copy:
                dest_file = os.path.join(dest, video.group + '_' + video.clip + '_' + f)
                if os.path.exists(dest_file):
                    continue

                shutil.copy2(os.path.join(frame_folder_name, f), dest_file)

            progress_bar.update(1)

        print('Next subset')
    progress_bar.close()
    print('END!')


if __name__ == '__main__':
    main()
