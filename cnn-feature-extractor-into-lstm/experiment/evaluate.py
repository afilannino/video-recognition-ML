import math
import os

from utility.experiment_utilities import retrieve_sequence, create_sequence_generator
from utility.utility import retrieve_videoobject_subsets, project_root, retrieve_classes
from experiment.train import create_final_model

project_root = project_root()
feature_length = 2048  # This is the length of the numpy array that contains the prediction
classes_size = 101  # Number of classes in UCF101 dataset
feature_sequence_size = 100  # This must be coherent with the value used in features_extractor.py


def main():
    batch_size = 64

    preload_features_in_memory = False

    model = load_final_model()
    classes = retrieve_classes()
    subsets = retrieve_videoobject_subsets(['test'])
    evaluate_model(model, classes, subsets, batch_size, preload_features_in_memory=preload_features_in_memory)


# This function retrieve only the first model_weights found in the folder. Change it if you want different behaviour
def load_final_model():
    model_weights_path = os.path.join(project_root, 'data', 'result', 'model_weights', 'PUT_HERE_WEIGHTS_FILENAME')
    return create_final_model(True, model_weights_path)


def evaluate_model(model, classes, subsets, batch_size, preload_features_in_memory=False):
    test_set = subsets[0]
    if preload_features_in_memory:
        x_test, y_test = retrieve_sequence(test_set, classes, feature_sequence_size, feature_length)
        score = model.evaluate(x_test, y_test, verbose=1)
        print('\nLoss: ', score[0])
        print('Accuracy: ', score[1])
        print('Top 5 categorical: ', score[2])

    else:
        test_sequence_generator = create_sequence_generator(test_set, classes, batch_size, feature_sequence_size,
                                                            feature_length)
        test_steps_per_epoch = math.ceil(len(test_set) / batch_size)

        score = model.evaluate_generator(test_sequence_generator,
                                         steps=test_steps_per_epoch,
                                         verbose=1)
        print('\nLoss: ', score[0])
        print('Accuracy: ', score[1])
        print('Top 5 categorical: ', score[2])

    print('END!')


if __name__ == '__main__':
    main()
