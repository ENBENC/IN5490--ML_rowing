import os
import sys
import pandas as pd
import tensorflow as tf
from tensorflow import keras

PROJECT_ROOT = os.path.abspath(os.path.join(
              os.path.dirname(__file__),
              os.pardir))
sys.path.append(PROJECT_ROOT)
import utils

file_path = os.path.dirname(os.path.realpath(__file__))

def save_metrics_to_file(feature_nr,loss,acc,use_angles,model_type,run_nr,test_set_type):
    if test_set_type == 2:
        model_type += "_2"

    outdir = os.path.join(file_path,"..","out/metrics")
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    filename = ""
    if test_set_type > 0:
        filename = outdir+"/"+model_type+"_coords_test.csv"
        if use_angles:
            filename = outdir+"/"+model_type+"_angles_test.csv"
    else:
        filename = outdir+"/"+model_type+"_coords_val.csv"
        if use_angles:
            filename = outdir+"/"+model_type+"_angles_val.csv"

    with open(filename,'a') as f:
        f.write(str(feature_nr)+","+str(run_nr)+","+str(loss)+","+str(acc)+"\n")

def main():
    args = sys.argv[1:]

    model_file_name = ""
    if len(args) > 0:
        model_file_name = args[0]
    else:
        print("Model filename missing. Should be path from home dir, e.g. out/train/angles_f2_run1/my_best_model.h5")
        exit()

    model_type = "gru"
    if len(args) > 1:
        if "lstm" in args[1]:
            model_type = "lstm"
    else:
        print("model type missing: 'gru' or 'lstm'")
        exit()

    use_angles = False
    if len(args) > 2:
        if "ang" in args[2]:
            use_angles = True
    else:
        print("feature type missing: 'angles' or 'coords'")
        exit()

    featur_nr = 0
    if len(args) > 3:
        try:
            feature_nr = int(args[3])
        except:
            print("feature number not an int")
            exit()
    else:
        print("feature number missing (int)")
        exit()

    run_nr = 0
    if len(args) > 4:
        try:
            run_nr = int(args[4])
        except:
            print("run number not an int")
            exit()
    else:
        print("run number missing (int)")
        exit()

    test_set_type = 0
    if len(args) > 5 and "test2" in args[5]:
        print("Using test set 2")
        test_set_type = 2
    elif len(args) > 5 and "test" in args[5]:
        print("Using test set 1")
        test_set_type = 1
    else:
        print("Using validation set. Write 'test' or 'test2' to use test set")

    # TensorBoard
    f_type = "angles" if use_angles else "coords"
    callback_filename = "logs/fit/val_"+model_type+"_"+f_type
    if test_set_type == 2:
        callback_filename = "logs/fit/test2_"+model_type+"_"+f_type
    elif test_set_type == 1:
        callback_filename = "logs/fit/test_"+model_type+"_"+f_type
    log_dir = os.path.join(file_path,"..",callback_filename)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    features = utils.xyz_features
    if use_angles:
        features = utils.angle_features

    # Load test set
    model_path = os.path.join(file_path,"..",model_file_name)
    model = keras.models.load_model(model_path)
    test_dataset, test_labels = utils.create_dataset(utils.val_files,features[0],use_angles)
    if test_set_type == 2:
            test_dataset, test_labels = utils.create_dataset(utils.test_files_2,features[0],use_angles)
    elif test_set_type == 1:
        test_dataset, test_labels = utils.create_dataset(utils.test_files,features[0],use_angles)
    x_test = tf.ragged.constant(test_dataset)
    y_test = tf.constant(test_labels)

    # Evaluate model
    loss, acc = model.evaluate(
        x=x_test,
        y=y_test,
        callbacks=[tensorboard_callback]
    )

    save_metrics_to_file(feature_nr,loss,acc,use_angles,model_type,run_nr,test_set_type)

if __name__ == "__main__":
    main()
