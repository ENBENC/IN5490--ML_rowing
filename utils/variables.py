import os
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf

"""
Variables used in utils.py and rnn_model_kristort
"""

# DEFINE CONSTANTS
timestep_size = 240 #Each line is 1/240 sek --> multiply with 240 instead of divide with 1/240
sequence_length = 160 #sek
nr_of_timesteps = timestep_size*sequence_length
seed = 42

xyz_features = np.array([
    #['lsh','rsh','lhi','rhi']
    #['lsh','rsh','lse','rse'],
    #['lsh','rsh','pfr','afr'],
    #['lha','rha','pfr','afr'],
    ['lhi','rhi','pfr','afr']
])
"""
xyz_features = np.array([['lsh','rsh','rhi','lhi','rse','lse','rha','mha','lha','pfr','mfr','afr']])
"""
angle_features = np.array([
    ['rsh','rhi','rha'], #Shoulder-angle
    #['rhi','rse','rsh'], #Hip-angle
    #['rse','lse','rhi'] #Seat-angle
])

file_path = os.path.dirname(os.path.realpath(__file__))
timedata = pd.read_csv(os.path.join(file_path,"../src/timedata_mocap.csv"), sep=",")

all_files = [
    "src/id1_phml_c4_mocap.csv", # Elite
    "src/id2_phml_c2_mocap.csv",
    "src/id3_phml_c6_mocap.csv",
    "src/id4_phml_c1_mocap.csv",
    "src/id5_phml_c6_mocap.csv",
    "src/id6_phml_c5_mocap.csv",
    "src/id7_phml_c4_mocap.csv",
    "src/id8_phml_c5_mocap.csv",
    "src/id9_phml_c1_mocap.csv", # Elite
    "src/id10_phml_c4_mocap.csv", # Non-elite
    "src/id11_phml_c2_mocap.csv",
    "src/id12_phml_c6_mocap.csv",
    "src/id13_phml_c1_mocap.csv",
    "src/id14_phml_c6_mocap.csv",
    "src/id15_phml_c5_mocap.csv",
    "src/id16_phml_c4_mocap.csv",
    "src/id17_phml_c5_mocap.csv",
    "src/id18_phml_c1_mocap.csv",
    "src/id19_phml_c5_mocap.csv",
    "src/id20_phml_c4_mocap.csv",
    "src/id21_phml_c4_mocap.csv", # Non-elite
    "src/id22_phml_c2_mocap.csv", # Elite
    "src/id23_phml_c6_mocap.csv", # Elite
]

train_files = [ #10-12 files: 5-6 elite, 5-6 non-elite
    all_files[0],
    all_files[1],
    all_files[2],
    all_files[5],
    all_files[6],
    all_files[9],
    all_files[10],
    all_files[12],
    all_files[13],
    all_files[19]
]

val_files = [ # 3-4 files
    all_files[4],
    all_files[8],
    all_files[16],
    all_files[17]
]

test_files = [ # 3-4 files
    all_files[3],
    all_files[7],
    all_files[11],
    all_files[18]
]

test_files_2 = [ # Less preprocessing done
    all_files[14], # Non-elite
    all_files[15], # Non-elite
    #all_files[20], # Non-elite
    all_files[21], # Elite
    all_files[22]  # Elite
]
