import glob
import os
import time
import pandas as pd
import tqdm
import csv

from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
# from keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing.image import ImageDataGenerator

from utility.utility import retrieve_classes, project_root, retrieve_videoobject_subsets, limit_frames_number

project_root = project_root()
classes = retrieve_classes()
flowframes_considered_per_video = 60
batch_size = 256
epochs = 50


def main():
    create_dataframe()

    base_model = InceptionV3(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D(name='final_avg_pool')(x)
    x = Dense(len(classes), activation='softmax', name='final_dense')(x)

    model = Model(inputs=base_model.input, outputs=x)

    # Hyper parameters
    loss = 'categorical_crossentropy'
    optimizer = Adam(lr=1e-5, decay=1e-6)
    metrics = ['accuracy', 'top_k_categorical_accuracy']

    # PART 1
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    train_generator = create_data_generator()

    model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=3
    )

    # PART 2
    for layer in model.layers[:249]:
        layer.trainable = False
    for layer in model.layers[249:]:
        layer.trainable = True
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    # Callback: function to save the model weights
    model_saver = ModelCheckpoint(
        filepath=os.path.join(project_root, 'data', 'result', 'model_weights',
                              'cnn-training-{epoch:03d}-{val_loss:.3f}.hdf5'),
        verbose=1,
        save_best_only=True)

    # Callback: function to log result
    timestamp = time.time()
    filelog_name = 'cnn-training-' + str(timestamp) + '.log'
    log_path = os.path.join(project_root, 'data', 'result', 'logs', filelog_name)
    csv_logger = CSVLogger(log_path)

    train_generator = create_data_generator()

    model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        callbacks=[model_saver, csv_logger]
    )

    print('Model trained!')


def create_data_generator():
    # https://stackoverflow.com/a/52372042/8094245
    img_height = 299
    img_width = 299
    # data_folder = os.path.join(project_root, 'data', 'UCF-101-cnn-frames')
    retraining_set = pd.read_csv(os.path.join(project_root, 'data', 'dataframe_retraining.csv'), delimiter=',')
    retraining_set.columns = ['FILENAME', 'LABEL']

    datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2)

    train_generator = datagen.flow_from_dataframe(
        retraining_set,
        None,
        x_col='FILENAME',
        y_col='LABEL',
        target_size=(img_height, img_width),
        class_mode='categorical',
        batch_size=batch_size)

    # train_generator = datagen.flow_from_directory(
    #     data_folder,
    #     target_size=(img_height, img_width),
    #     batch_size=batch_size,
    #     classes=classes,
    #     class_mode='categorical')

    return train_generator


def create_dataframe():
    subsets = retrieve_videoobject_subsets(['train'])
    total_frames_list = []

    # Initializing progress bar
    length = 0
    for videoobject_subset in subsets:
        length += len(videoobject_subset)
    print('Starting dataframe creation for image_generator')
    progress_bar = tqdm.tqdm(total=length)

    # Starting looping over subset of videos
    for videoobject_subset in subsets:

        for video in videoobject_subset:
            frames_folder = os.path.join(project_root, 'data', 'UCF-101-frames', video.label,
                                      'v_' + video.label + '_' + video.group + '_' + video.clip) + '_frames'
            flowframes_folder = os.path.join(project_root, '/additional-storage', 'UCF-101-flowframes', video.label,
                                      'v_' + video.label + '_' + video.group + '_' + video.clip) + '_flowframes'

            if not os.path.exists(frames_folder):
                os.makedirs(frames_folder)
            if not os.path.exists(flowframes_folder):
                os.makedirs(flowframes_folder)

            label = video.label

            frames_list = glob.glob(os.path.join(frames_folder, '*'))
            frames_list = limit_frames_number(frames_list, flowframes_considered_per_video)
            for frame in frames_list:
                total_frames_list.append((frame, label))

            flowframes_list = glob.glob(os.path.join(flowframes_folder, '*'))
            flowframes_list = limit_frames_number(flowframes_list, flowframes_considered_per_video)
            for frame in flowframes_list:
                total_frames_list.append((frame, label))
            progress_bar.update(1)

    progress_bar.close()

    header = ['FILENAME', 'LABEL']
    with open(os.path.join(project_root, 'data', 'dataframe_retraining.csv'), 'w', newline='\n') as retrain_csv:
        retrain_csv_writer = csv.writer(retrain_csv, delimiter=',')
        retrain_csv_writer.writerow(header)
        for entry in total_frames_list:
            retrain_csv_writer.writerow(entry)
    print('CSV with frames filename correctly created')


if __name__ == '__main__':
    main()
