import shutil
import os

from utility.utility import project_root, retrieve_classes

project_root = project_root()


def main():
    move_features()


def move_features():

    for classs in retrieve_classes():
        source = os.path.join(project_root, '..', 'lstm-with-cnn-features-only-rgb', 'data', 'UCF-101', classs, classs + '_features')
        files = os.listdir(source)

        dest = os.path.join(project_root, 'data', 'UCF-101', classs, classs + '_features')
        print('DESTINATION: ' + dest)
        if not os.path.exists(dest):
            os.makedirs(dest)

        for f in files:
            shutil.copy2(os.path.join(source, f), dest)
            print('SOURCE: ' + source + f)

        print("End class: " + classs)

    print('Done!')


if __name__ == '__main__':
    main()
