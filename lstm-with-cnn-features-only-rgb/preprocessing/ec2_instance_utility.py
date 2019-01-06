import shutil
import os

from utility.utility import project_root, retrieve_videoobject_subsets

project_root = project_root()


def main():
    move_features()


def move_features():
    for videoobject_subset in retrieve_videoobject_subsets(['train', 'validation']):

        for video in videoobject_subset:
            video_base = os.path.join(project_root, 'data', 'UCF-101', video.label,
                                      'v_' + video.label + '_' + video.group + '_' + video.clip)

            source = os.path.join(project_root, 'data', 'exp1_rgbfeatures_20', video.label, video.label + '_features')
            files = os.listdir(source)

            dest = os.path.join(project_root, 'data', 'UCF-101', video.label, video.label + '_features')

            if not os.path.exists(dest):
                os.makedirs(dest)

            for f in files:
                shutil.move(os.path.join(source, f), dest)

    print('Done')


if __name__ == '__main__':
    main()
