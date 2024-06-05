import os
import gdown
import zipfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from hyperparameter import parameter


def download_dataset():
    path = 'data.zip'
    if not os.path.isfile(path):
        ID = "1_AziIzy6iCUVzQewPlq_QhBvdXlsmyg_"
        gdown.download(id=ID, output=path, quiet=False)
    else:
        print("Dataset cache found")


def extract_dataset():
    if not os.path.isdir('data'):
        print("Extracting dataset...")
        zip_ref = zipfile.ZipFile('data.zip', 'r')
        zip_ref.extractall()
        zip_ref.close()


def get_datasaet():
    download_dataset()
    extract_dataset()

    train_dir = 'data/train'
    val_dir = 'data/valid'
    test_dir = 'data/test'

    train_datagen = ImageDataGenerator(
        rescale=1 / 255.,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    val_datagen = ImageDataGenerator(rescale=1 / 255.)
    test_datagen = ImageDataGenerator(rescale=1 / 255.)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=parameter['image_shape'],
        batch_size=parameter['training_batch_size'],
        class_mode='categorical',
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=parameter['image_shape'],
        batch_size=parameter['validation_batch_size'],
        class_mode='categorical',
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=parameter['image_shape'],
        batch_size=parameter['test_batch_size'],
        class_mode='categorical',
    )

    return train_generator, val_generator, test_generator
