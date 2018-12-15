import os
import csv
import random
from utility.video_object import VideoObject
from utility.utility import project_root, ffmpeg_path

project_root = project_root()
ffmpeg_path = ffmpeg_path()

# Official UCF101 has 3 different splitting modes
split_version = '01'  # Possible values are 01, 02 or 03


def main():
    video_list = retrieve_dataset()
    # split_dataset(video_list)
    retrieve_official_split(split=split_version)
    generate_classes(video_list)


def retrieve_official_split(split='01'):

    # Check paths existence
    testfile = os.path.join(project_root, 'data', 'ucfTrainTestlist', 'testlist' + split + '.txt')
    trainfile = os.path.join(project_root, 'data', 'ucfTrainTestlist', 'trainlist' + split + '.txt')
    if not os.path.exists(testfile):
        raise Exception('File not found: ' + testfile)
    if not os.path.exists(trainfile):
        raise Exception('File not found: ' + trainfile)

    with open(testfile) as f1:
        testlist = f1.readlines()
    with open(trainfile) as f2:
        trainlist = f2.readlines()

    test_set = list(map(retrieve_videoobject_from_string, testlist))
    train_set = list(map(retrieve_videoobject_from_string, trainlist))

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
        for video in test_set:
            validation_csv_writer.writerow([video.label, video.group, video.clip])


def retrieve_dataset():
    # Check paths existence
    testfile = os.path.join(project_root, 'data', 'ucfTrainTestlist', 'testlist' + split_version + '.txt')
    trainfile = os.path.join(project_root, 'data', 'ucfTrainTestlist', 'trainlist' + split_version + '.txt')
    if not os.path.exists(testfile):
        raise Exception('File not found: ' + testfile)
    if not os.path.exists(trainfile):
        raise Exception('File not found: ' + trainfile)

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


# Function that retrieve VideoObject from a string
def retrieve_videoobject_from_string(entry_string):
    splits = entry_string.split('_')
    return VideoObject(splits[1], splits[2], splits[3][:3])


def split_dataset(video_list):
    train_set = []
    validation_set = []
    test_set = []

    # Retrieve and sort the classes
    classes = list(set(map(lambda x: x.label, video_list)))  # set is used to obtain distinct element
    classes.sort()

    for label in classes:
        # Retrieve subset of video for this label
        subset = list(filter(lambda x: x.label == label, video_list))

        # Generate a group division for test (0.25) validation (0.10) and train (0.65)
        groups_split = list(range(25))
        random.shuffle(groups_split)
        size = len(groups_split)
        # size1 = int(size * 0.65)
        # size2 = int(size * 0.75)
        train_groups = groups_split[:int(size * 0.65)]
        validation_groups = groups_split[int(size * 0.65): int(size * 0.75)]
        test_groups = groups_split[int(size * 0.75):]

        for group in train_groups:
            train_set += list(filter(lambda x: x.group == 'g{:02d}'.format(group+1), subset))

        for group in validation_groups:
            validation_set += list(filter(lambda x: x.group == 'g{:02d}'.format(group+1), subset))

        for group in test_groups:
            test_set += list(filter(lambda x: x.group == 'g{:02d}'.format(group+1), subset))

    print('Video splitted: ', len(train_set) + len(validation_set) + len(test_set))

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


def generate_classes(video_list):
    classes = list(set(map(lambda video: video.label, video_list)))  # set is used to obtain distinct element
    classes.sort()

    with open(os.path.join(project_root, 'data', 'classes.csv'), 'w', newline='\n') as classes_csv:
        classes_csv_writer = csv.writer(classes_csv, delimiter=',')
        for label in classes:
            classes_csv_writer.writerow([label])
    return


if __name__ == '__main__':
    main()
