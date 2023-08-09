import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(
              os.path.dirname(__file__),
              os.pardir))
sys.path.append(PROJECT_ROOT)

from utils import all_files, filename_to_separator, count_data_loss


# For the new files
"""
from utils import count_data_loss

def filename_to_separator(x):
    return ","

all_files = [
    "src/id1_phml_c4_mocap.csv",
    "src/id2_phml_c2_mocap.csv",
    "src/id3_phml_c6_mocap.csv",
    "src/id4_phml_c1_mocap.csv",
    "src/id5_phml_c6_mocap.csv",
    "src/id6_phml_c5_mocap.csv",
    "src/id7_phml_c4_mocap.csv",
    "src/id8_phml_c5_mocap.csv",
    "src/id9_phml_c1_mocap.csv",
    "src/id10_phml_c4_mocap.csv",
    "src/id11_phml_c2_mocap.csv",
    "src/id12_phml_c6_mocap.csv",
    "src/id13_phml_c1_mocap.csv",
    "src/id14_phml_c6_mocap.csv",
    "src/id17_phml_c5_mocap.csv",
    "src/id18_phml_c1_mocap.csv",
    "src/id19_phml_c5_mocap.csv",
    "src/id20_phml_c4_mocap.csv"
]

with open(os.path.join(file_path,'../data_loss_new_files.txt'), 'w') as output_file:
"""

def main():
    file_path = os.path.dirname(os.path.realpath(__file__))

    with open(os.path.join(file_path,'../data_loss.txt'), 'w') as output_file:

        write_header = False
        for file in all_files:
            filename = os.path.join(file_path,"..",file)
            data = pd.read_csv(filename,sep=filename_to_separator(filename))
            df = pd.DataFrame(data)
            column_names,data_loss = count_data_loss(df,filename)

            if not write_header:
                output_file.write("filename,"+",".join(column_names)+"\n")
                write_header = True

            output_file.write(file.split("/")[-1]+","+str(data_loss).replace("]","").replace("[","").replace(" ","")+"\n")

    output_file.close()

if __name__ == "__main__":
    main()
