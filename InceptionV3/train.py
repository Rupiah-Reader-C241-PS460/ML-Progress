import os
import data
from model import Model
from tensorflow.keras.optimizers import SGD
from hyperparameter import parameter


def save_model(model):
    version = parameter['version']
    export_path = os.path.join("saved_model", str(version))

    if os.path.isdir(export_path):
        print("\nClean up previous model")
        os.remove(export_path)

    model.save(export_path, save_format="tf")


if __name__ == "__main__":
    train_generator, val_generator, test_generator = data.get_datasaet()

    model = Model(7)
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(learning_rate=parameter['learning_rate'], momentum=0.9),
                  metrics=['accuracy'])

    model.fit(train_generator, epochs=parameter['epochs'], validation_data=val_generator)
    model.evaluate(test_generator)

    save_model(model)
