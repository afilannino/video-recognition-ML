"""
This script is used to split dataset and extract frames from dataset's video
"""
import csv
import os
import os.path
import random
import subprocess

import tqdm

from .video_object import VideoObject

project_root = 'INSERIRE PATH ROOT PROGETTO'
ffmpeg_path = 'ffmpeg'  # Specificare se diverso


def main():
    video_list = retrieve_dataset()
    split_dataset(video_list)
    generate_classes(video_list)
    generate_frames(['train', 'validation', 'test'])


def retrieve_dataset():
    # Official UCF101 has 3 different splitting modes
    split_version = '01'  # Possible values are 01, 02 or 03

    # Check paths existence
    testfile = os.path.join(project_root, 'data', 'ucfTrainTestlist', 'testlist' + split_version + '.txt')
    trainfile = os.path.join(project_root, 'data', 'ucfTrainTestlist', 'trainlist' + split_version + '.txt')
    if not os.path.exists(testfile):
        print('Incorrect path for file: ' + testfile)
        return -1
    if not os.path.exists(trainfile):
        print('Incorrect path for file: ' + trainfile)
        return -1

    # Obtain all file lines as an array
    datalist = []
    with open(testfile) as f1:
        datalist += f1.readlines()
    with open(trainfile) as f2:
        datalist += f2.readlines()

    datalist.sort()
    print('Number of video: ', len(datalist))
    # print(dataset)

    # Return an array of VideoObject
    return list(map(retrieve_videoobject_from_string, datalist))


# Create a function that retrive VideoObject from a string
def retrieve_videoobject_from_string(entry_string):
    splits = entry_string.split('_')
    return VideoObject(splits[1], splits[2], splits[3][:3])


def generate_classes(video_list):
    classes = list(set(map(lambda video: video.label, video_list)))  # set is used to obtain distinct element
    classes.sort()

    with open(os.path.join(project_root, 'data', 'classes.csv'), 'w', newline='\n') as classes_csv:
        classes_csv_writer = csv.writer(classes_csv, delimiter=',')
        for label in classes:
            classes_csv_writer.writerow([label])
    return


def split_dataset(video_list):
    train_set = []
    validation_set = []
    test_set = []

    # Retrieve and sort the classes
    classes = list(set(map(lambda video: video.label, video_list)))  # set is used to obtain distinct element
    classes.sort()

    for label in classes:
        # Retrieve subset of video for this label
        subset = list(filter(lambda video: video.label == label, video_list))

        # Generate a group division for test (0.25) validation (0.10) and train (0.65)
        groups_split = list(range(25))
        random.shuffle(groups_split)
        size = len(groups_split)

        for group in groups_split[: int(size * 0.65)]:
            train_set += list(filter(lambda x: x.group == 'g{:02d}'.format(group), subset))

        for group in groups_split[int(size * 0.65): int(size * 0.75)]:
            validation_set += list(filter(lambda x: x.group == 'g{:02d}'.format(group), subset))

        for group in groups_split[int(size * 0.75):]:
            test_set += list(filter(lambda x: x.group == 'g{:02d}'.format(group), subset))

    # Write the subset generated into dedicated csv
    header = ['LABEL', 'GROUP', 'CLIP']
    with open(os.path.join(project_root, 'data', 'train-set.csv'), 'w', newline='\n') as train_csv:
        train_csv_writer = csv.writer(train_csv, delimiter=',')
        train_csv_writer.writerow(header)
        for video in train_set:
            train_csv_writer.writerow([video.label, video.group, video.clip])

    with open(os.path.join(project_root, 'data', 'validation-set.csv'), 'w', newline='\n') as validation_csv:
        validation_csv_writer = csv.writer(validation_csv, delimiter=',')
        validation_csv_writer.writerow(header)
        for video in validation_set:
            validation_csv_writer.writerow([video.label, video.group, video.clip])

    with open(os.path.join(project_root, 'data', 'test-set.csv'), 'w', newline='\n') as test_csv:
        test_csv_writer = csv.writer(test_csv, delimiter=',')
        test_csv_writer.writerow(header)
        for video in test_set:
            test_csv_writer.writerow([video.label, video.group, video.clip])


def retrieve_videoobject_subsets(subsets):
    videoobject_subsets = []
    # Loop over the three subsets: train, validation, test
    for subset in subsets:
        videoobject_subset = []
        # Open the csv file and retrieve a list of VideoObject
        with open(os.path.join(project_root, 'data', subset + '-set.csv'), 'r', newline='\n') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            next(csv_reader)  # Skip the header
            for row in csv_reader:
                videoobject_subset.append(VideoObject(row[0], row[1], row[2]))
        videoobject_subsets.append(videoobject_subset)
    return videoobject_subsets


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


if __name__ == '__main__':
    main()
