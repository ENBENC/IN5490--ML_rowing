import os
import math
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from .variables import timedata, timestep_size, seed

"""
Fuctions used in rnn_model_kristort

Testing: run 'pytest test_rnn_model.py'
"""

def to_xyz_features(data):
    """
        Turns a flattened dataset (with respect to features) into a dataset with coordinate vectors

        Args:
            data (numpy.ndarray NxM): dataset with
                N time steps (e.g. 38400) and
                M feature markers (e.g. 36)

        Returns:
            features (numpy.ndarray NxKx3)
                K = M/3
    """

    vector_size = 3
    nr_of_features = data.shape[1]//vector_size
    features = np.zeros((data.shape[0],nr_of_features,vector_size))

    feature_index = 0
    vector_index = 0

    # For each timestep ...
    for time_step_index in range(data.shape[0]):
        # ... group the 36 values into 12 xyz-vectors
        for value in data[time_step_index]:
            # Save the xyz-vector to the feature array
            features[time_step_index,feature_index,vector_index] = value
            vector_index = (vector_index+1) % vector_size

            if vector_index % vector_size == 0:
                feature_index = (feature_index+1) % nr_of_features

    return features

#def split_data(data, chunk, label):
#    return np.array_split(data,chunk), [label for i in range(chunk)]

def split_data(data,chunk_length,chunk_overlap,label):
    step_size = chunk_length-chunk_overlap
    number_of_chunks = int(np.ceil((data.shape[0]-chunk_overlap)/step_size))
    new_data = np.zeros((number_of_chunks,chunk_length,*data.shape[1:]))

    new_labels = np.zeros((number_of_chunks))
    if label == 1:
        new_labels = np.ones((number_of_chunks))

    index = 0
    for i in range(0,data.shape[0],step_size):
        if index < new_data.shape[0]:
            if data[i:i+chunk_length].shape != new_data[index].shape:
                for j,elem in enumerate(data[i:i+chunk_length]):
                    new_data[index][j] = elem
            else:
                new_data[index] = data[i:i+chunk_length]
            index += 1

    return new_data,new_labels

def filename_to_separator(filename):
    if 'src/' in filename:
        filename = filename.split("src/")[1]

    id = int(filename.split("_")[0].split("id")[1])

    if id == 7 or id == 9 or id == 10 or id == 11:
        return ";"
    return ","

def normalize(x):
    """
        Args:
            x (numpy.ndarray): Array to be normalized

        Returns:
            (numpy.ndarray): Normalized array
    """
    min = np.min(x)
    max = np.max(x)
    return (x-min)/(max-min)

def create_inp_vectors(file,start,stop,label,features,use_angles):
    """
        Reads from file path and creates input vectors from the data
        Splicing of start and stop time

        Args:
            file (str)  : file name path
            start (int) : start index row
            stop (int)  : stop index row

        Returns:
            x (list) : input vector
    """

    # Load file into a pandas DataFrames
    data = pd.read_csv(file,sep=filename_to_separator(file))
    # Create a dataframe
    df = pd.DataFrame(data)

    # Removes row and time columns
    df.pop('row')
    df.pop('time')

    xyz_features = []
    for feature in features:
        xyz_features.append(feature+"_x")
        xyz_features.append(feature+"_y")
        xyz_features.append(feature+"_z")

    # Use only features: lsh_x, rsh_x
    #print(df.head())
    df = df[xyz_features]
    #df = df[['lsh_x','rsh_x','rhi_x','lhi_x','rse_x']]
    #print(df.head())

    x = df[start:stop]

    if use_angles:
        x = np.array(data_to_angle(x,[features]))
    else:
        x = x.to_numpy()
        x = normalize(x) #tf.keras.utils.normalize(data)
        #x = to_xyz_features(x)

    median_stroke_length_sek = get_median_stroke_length(file)
    error_margin_sek = 0.25
    stroke_length_timesteps = int((median_stroke_length_sek+error_margin_sek)*timestep_size)
    stroke_overlap_sek = 1
    stroke_overlap_timesteps = int(stroke_overlap_sek*timestep_size)
    x,y = split_data(x,stroke_length_timesteps,stroke_overlap_timesteps,label)

    x = np.array(x).tolist()

    return x, y

def file_to_dataset(file,features,use_angles):
    def find_start_stop(file):
        if 'src/' in file:
            file = file.split("src/")[1]
        row = timedata.loc[timedata['Filename'] == file]
        start = int((row['Start']+10)*timestep_size)
        stop = int((row['Stop']-10)*timestep_size)
        return start, stop

    start, stop = find_start_stop(file)
    label = filename_to_label(file)
    return create_inp_vectors(file,start,stop,label,features,use_angles)

def create_dataset(files,features,use_angles=False):
    dataset,labels = [],[]

    file_path = os.path.dirname(os.path.realpath(__file__))

    for file in files:
        file = os.path.join(file_path,"..",file)
        x,y = file_to_dataset(file,features,use_angles)

        for i in range(len(x)):
            dataset.append(x[i])
            labels.append(y[i])

    # Shuffle datapoints
    dataset,labels = shuffle(dataset,labels,random_state=seed)

    return dataset,labels

def filename_to_id(filename):
    if 'src/' in filename:
        filename = filename.split("src/")[1]
    id_str = timedata.loc[timedata['Filename'] == filename]['ID']
    return int(id_str)

def filename_to_label(filename):
    """
        Args:
            filename(str): Name of file

        Returns:
            label (int): 0 (non-elite) or 1 (elite)
    """
    id = filename_to_id(filename)
    return int(id < 10 or id > 21)

def get_median_stroke_length(rowing_file):
    filename = rowing_file
    if "src/" in filename:
        filename = rowing_file.split("src/")[1]

    median_stroke = 0
    file_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(file_path,'../median_strokes.txt'), 'r') as strokes_file:

        for line in strokes_file.readlines():
            stroke_filename,stroke_length = line.split(",")
            if filename == stroke_filename:
                median_stroke = float(stroke_length)
                break

    strokes_file.close()

    return median_stroke

def angle_between_two_line(A,B,C):
    """
    Take 3 points, where each point is A=(x,y,z)

    Args:
        A(np array): a point
        B(np array): a point
        C(np array):a point

    Returns:
        radiens(float): radian bewteen vector AB and AC
    """
    #Find vector
    a = C - A
    b = B - A
    #tmp must be in range of -1 to 1 to find acos of it
    tmp = np.dot(a,b)
    if tmp != 0:
        tmp /= np.sqrt(np.dot(a,a))*np.sqrt(np.dot(b,b))
    if tmp > 1 or tmp < -1:
        tmp = round(tmp)

    radians = math.acos(tmp)
    return radians

def get_angle_array(point1_vec, point2_vec, point3_vec):
    """
    Ex: point1_vec = [A_timestep1
                    A_timestep2
                    ...
                    A_timestepN]

    Take 3 point_vector that include alle point in a sequence of timestep

    Args:
        point1_vec(np array): alle point in a sequence of timestep
        point2_vec(np array): alle point in a sequence of timestep
        point3_vec(np array): alle point in a sequence of timestep

    Returns:
        lst(np array): array with radians

    """
    lst = []
    for i in range(len(point1_vec)):
        lst.append(angle_between_two_line(point1_vec[i], point2_vec[i], point3_vec[i]))
    return np.array(lst)

def data_to_angle(df, lst_3_features):
    """
    Take a list with label of feature in DataFrame df example ['feature1', 'feature2', feature3]

    Args:
        df(pandas DataFrame) : a dataframe that contains all feature to a person
        lst_3_features(python list) : list of 3 features

    Returns:
        angle_lst(np array): array with radians
    """

    all_xyz_features = []
    for i in range(len(lst_3_features)):
        xyz_features = []
        for feature in lst_3_features[i]:
            #Create all 9 column label
            xyz_feature = np.array([df[feature+"_x"],df[feature+"_y"],df[feature+"_z"]])
            xyz_features.append(xyz_feature.T)
        all_xyz_features.append(xyz_features)

    #Get all 3 points vector
    nr_of_timesteps = df.shape[0]
    nr_of_angles = len(lst_3_features)

    all_angles = np.zeros((nr_of_timesteps,nr_of_angles))
    for i in range(len(all_xyz_features)):
        feature = all_xyz_features[i]
        angle_timesteps = get_angle_array(feature[0], feature[1], feature[2])
        all_angles[:,i] = angle_timesteps.T

    return all_angles


def handle_names_are_wrong(df):
    df.replace(0, np.nan, inplace=True)

    rha_mean = df['rha_y'].mean()
    lha_mean = df['lha_y'].mean()
    rsh_mean = df['rsh_y'].mean()

    distance_rsh_to_rha = np.abs(rha_mean - rsh_mean)
    distance_rsh_to_lha = np.abs(lha_mean - rsh_mean)

    if distance_rsh_to_rha > distance_rsh_to_lha:
        return True
    return False

def fix_handle_names(df,filename=None):
    df = df.copy(deep=True)
    change = handle_names_are_wrong(df)

    if change:
        temp = df['lha_y']
        df['lha_y'] = df['rha_y']
        df['rha_y'] = temp

        if filename:
            print("File",filename,"changed")
            df.to_csv(filename,sep=filename_to_separator(filename),index=False)
        else:
            print("Dataframe updated")

    return df

def count_data_loss(df,filename=None):
    data_loss = []
    columns = df.columns[2:].to_list()
    assert len(columns) == 36

    unique_columns = [i.split("_")[0] for i in columns]
    column_names = list(dict.fromkeys(unique_columns))
    assert len(column_names) == 12

    for column_name in column_names:
        data = (df[column_name+"_x"] == 0) & (df[column_name+"_y"] == 0) & (df[column_name+"_z"] == 0)
        data_loss.append(data.sum())

    return column_names,data_loss
