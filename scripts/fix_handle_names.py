import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(
              os.path.dirname(__file__),
              os.pardir))
sys.path.append(PROJECT_ROOT)

from utils import all_files, filename_to_separator, fix_handle_names

def main():
    file_path = os.path.dirname(os.path.realpath(__file__))

    for file in all_files:
        filename = os.path.join(file_path,"..",file)
        data = pd.read_csv(filename,sep=filename_to_separator(filename))
        df = pd.DataFrame(data)
        fix_handle_names(df,filename)

if __name__ == "__main__":
    main()
