import tqdm
import os
from shutil import move

from utility.utility import project_root, retrieve_videoobject_subsets

project_root = project_root()


def main():
    # Initializing progress bar
    length = 0
    for videoobject_subset in retrieve_videoobject_subsets(['train', 'validation']):
        length += len(videoobject_subset)
    progress_bar = tqdm.tqdm(total=length)

    for videoobject_subset in retrieve_videoobject_subsets(['train', 'validation']):

        for video in videoobject_subset:
            frame_folder_name = os.path.join(project_root, 'data', 'UCF-101-cnn-flowframes', video.label)

            dest = os.path.join(project_root, 'data', 'UCF-101-cnn-frames', video.label)

            files_to_move = os.listdir(frame_folder_name)

            if not os.path.exists(dest):
                os.makedirs(dest)

            for f in files_to_move:
                # print('\nSOURCE = ' + os.path.join(frame_folder_name, f))
                # print('DEST = ' + os.path.join(dest, video.group + '_' + video.clip + '_' + f))
                move(os.path.join(frame_folder_name, f), os.path.join(dest, video.group + '_' + video.clip + '_' + f))

            progress_bar.update(1)

    progress_bar.close()
    print('Done')


if __name__ == '__main__':
    main()
