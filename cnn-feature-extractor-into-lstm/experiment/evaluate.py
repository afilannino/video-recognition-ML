import glob
import os

from keras.models import load_model

from utility.experiment_utilities import retrieve_sequence, create_sequence_generator
from utility.utility import retrieve_videoobject_subsets, project_root, retrieve_classes

project_root = project_root()
feature_length = 2048  # This is the length of the numpy array that contains the prediction
classes_size = 101  # Number of classes in UCF101 dataset
feature_sequence_size = 100  # This must be coherent with the value used in features_extractor.py


def main():
    preload_features_in_memory = False

    model = load_final_model()
    classes = retrieve_classes()
    subsets = retrieve_videoobject_subsets(['test'])
    evaluate_model(model, classes, subsets, preload_features_in_memory=preload_features_in_memory)


# This function retrieve only the first model_weights found in the folder. Change it if you want different behaviour
def load_final_model():
    model_weights_path = os.path.join(project_root, 'data', 'result', 'model_weights')
    model_weights_filename = glob.glob(os.path.join(model_weights_path, 'finalmodel-*.hdf5'))
    return load_model(model_weights_filename[0])


def evaluate_model(model, classes, subsets, preload_features_in_memory=False):
    test_data = subsets[0]
    if preload_features_in_memory:
        x_test, y_test = retrieve_sequence(test_data, classes)
        score = model.evaluate(x_test, y_test, verbose=1)
        print('Loss: ', score[0])
        print('Accuracy: ', score[1])

    else:
        batch_size = 128
        train_steps_per_epoch = int(len(test_data) // batch_size)
        test_sequence_generator = create_sequence_generator(test_data, classes, batch_size)
        score = model.evaluate_generator(test_sequence_generator, train_steps_per_epoch, verbose=1)
        print(score)


if __name__ == '__main__':
    main()
