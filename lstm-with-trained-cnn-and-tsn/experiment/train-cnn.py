import os
import time

from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

from experiment.plot_data import main as create_plot
from utility.utility import retrieve_classes, project_root

project_root = project_root()
classes = retrieve_classes()
batch_size = 64


def main():
    base_model = InceptionV3(weights='imagenet', include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D(name='final_avg_pool')(x)
    x = Dense(1024, activation='relu', name='final_dense')(x)

    predictions = Dense(len(classes), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    train_generator, validation_generator = create_data_generator()

    model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        epochs=5
    )

    for layer in model.layers[:249]:
        layer.trainable = False
    for layer in model.layers[249:]:
        layer.trainable = True

    from keras.optimizers import SGD
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

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

    model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        epochs=200,
        callbacks=[model_saver, csv_logger]
    )
    create_plot(filelog_name)
    print('Model trained!')


def create_data_generator():
    # https://stackoverflow.com/a/52372042/8094245
    img_height = 299
    img_width = 299
    data_folder = os.path.join(project_root, 'data', 'UCF-101-cnn-frames')

    datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.3)

    train_generator = datagen.flow_from_directory(
        data_folder,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        classes=classes,
        class_mode='categorical',
        subset='training')

    validation_generator = datagen.flow_from_directory(
        data_folder,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        classes=classes,
        class_mode='categorical',
        subset='validation')

    return train_generator, validation_generator


if __name__ == '__main__':
    main()
