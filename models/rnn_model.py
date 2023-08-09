import os
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses, metrics
from keras.callbacks import ModelCheckpoint

PROJECT_ROOT = os.path.abspath(os.path.join(
              os.path.dirname(__file__),
              os.pardir))
sys.path.append(PROJECT_ROOT)
import utils

def main():
    batch_size = 4
    epochs = 15
    features = utils.xyz_features

    model_type = "gru"
    unit_size = 64
    if len(sys.argv[1:]) > 0 and sys.argv[1] == "rnn":
        model_type = "rnn"
        unit_size = 8
        print("Using simpleRNN")
    elif len(sys.argv[1:]) > 0 and sys.argv[1] == "lstm":
        model_type = "lstm"
        unit_size = 8
        print("Using LSTM")
    else:
        print("Using GRU")

    use_angles = False
    if len(sys.argv[1:]) > 1 and "ang" in sys.argv[2]:
        use_angles = True
        features = utils.angle_features
        print("Using angle-features")
    else:
        print("Using xyz-features")

    filename_number = 0
    if len(sys.argv[1:]) > 2:
        try:
            filename_number = int(sys.argv[3])
        except:
            filename_number = 0



    # MAKING THE DATASET
    for i in range(features.shape[0]):
        dataset, labels = utils.create_dataset(utils.train_files,features[i],use_angles)
        x_train = tf.ragged.constant(dataset)
        y_train = tf.constant(labels)
        val_dataset, val_labels = utils.create_dataset(utils.val_files,features[i],use_angles)
        x_val = tf.ragged.constant(val_dataset)
        y_val = tf.constant(val_labels)

        file_path = os.path.dirname(os.path.realpath(__file__))

        # Save best
        cp_filename = "my_best_"+model_type+"_model.epoch{epoch:02d}-acc{val_binary_accuracy:.2f}.hdf5"
        filepath = os.path.join(file_path,"..","out/train",cp_filename)
        checkpoint = ModelCheckpoint(
            filepath=filepath,
            monitor='val_binary_accuracy',
            verbose=1,
            save_best_only=True,
            mode='max')

        # Tensorboard
        file_path = os.path.dirname(os.path.realpath(__file__))
        f_type = "angles" if use_angles else "coords"
        callback_filename = "logs/fit/"+model_type+"_"+f_type+"_unitSize="+str(unit_size)
        if filename_number > 0:
            callback_filename += "_run="+str(filename_number)
        log_dir = os.path.join(file_path,"..",callback_filename)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        # CREATING THE MODEL
        nr_of_features = len(dataset[0][0])
        model = keras.Sequential()
        model.add(keras.Input(shape=(utils.nr_of_timesteps,nr_of_features)))

        if model_type == "gru":
            model.add(layers.GRU(unit_size, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
            model.add(layers.GRU(unit_size, dropout=0.2, recurrent_dropout=0.2))
        elif model_type == "lstm":
            model.add(layers.LSTM(unit_size, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
            model.add(layers.LSTM(unit_size, dropout=0.2, recurrent_dropout=0.2))
        else:
            model.add(layers.SimpleRNN(unit_size, activation='tanh'))

        model.add(layers.Dense(1, activation="sigmoid"))
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
            callbacks=[tensorboard_callback,checkpoint]
        )

        # Save the trained model
        model_path = os.path.join(file_path,"..","out/train", model_type+"_model_"+f_type+"_e"+str(epochs)+"_r"+str(filename_number)+".h5")
        model.save(model_path)

if __name__ == "__main__":
    main()
