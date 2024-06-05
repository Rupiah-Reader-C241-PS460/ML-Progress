import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from hyperparameter import parameter


class Model(tf.keras.Model):
    def __init__(self, num_classes=7):
        super(Model, self).__init__()
        self.inception = InceptionV3(
            weights='imagenet',
            include_top=False,
            input_shape=parameter['image_shape'] + (3,)
        )

        for layer in self.inception.layers:
            layer.trainable = False

        self.global_avg_pool = GlobalAveragePooling2D()
        self.dense = Dense(1024, activation='relu')
        self.dropout = Dropout(0.5)
        self.classifier = Dense(num_classes, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        x = self.inception(inputs)
        x = self.global_avg_pool(x)
        x = self.dense(x)
        x = self.dropout(x)
        return self.classifier(x)
