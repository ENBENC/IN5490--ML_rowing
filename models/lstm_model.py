import os
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses, metrics

PROJECT_ROOT = os.path.abspath(os.path.join(
              os.path.dirname(__file__),
              os.pardir))
sys.path.append(PROJECT_ROOT)
import utils

def main():
    batch_size = 1
    epochs = 15

    # MAKING THE DATASET
    for i in range(utils.xyz_features.shape[0]):
        dataset, labels = utils.create_dataset(utils.train_files, utils.xyz_features[i])
        x_train = tf.ragged.constant(dataset)
        y_train = tf.constant(labels)
        val_dataset, val_labels = utils.create_dataset(utils.val_files, utils.xyz_features[i])
        x_val = tf.ragged.constant(val_dataset)
        y_val = tf.constant(val_labels)

        nr_of_features = len(dataset[0][0])
        model = keras.Sequential()
        model.add(keras.Input(shape=(utils.nr_of_timesteps,nr_of_features)))
        model.add(layers.LSTM(2))
        model.add(layers.Dense(1))

        model.summary()


        # TRAINING THE MODEL with the compile-fit approach
        model.compile(
            optimizer="adam",
            loss=losses.BinaryCrossentropy(),
            metrics=metrics.BinaryAccuracy()
        )

        history = model.fit(
            x=x_train,
            y=y_train,
            validation_data=(x_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[utils.tensorboard_callback]
        )

        file_path = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(file_path,"..","out/train", "lstm_model"+str(i+1)+"_e"+str(epochs)+".h5")
        model.save(model_path)



if __name__ == "__main__":
    main()
