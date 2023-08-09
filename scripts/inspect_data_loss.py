import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(
              os.path.dirname(__file__),
              os.pardir))
sys.path.append(PROJECT_ROOT)

def main():
    file_path = os.path.dirname(os.path.realpath(__file__))

    loss_file1 = os.path.join(file_path,'../data_loss.txt')
    loss_file2 = os.path.join(file_path,'../data_loss_new_files.txt')

    df1 = pd.read_csv(loss_file1)
    df2 = pd.read_csv(loss_file2)
    df = pd.merge(df1, df2, how ='inner', on='filename')

    print("Loss difference (over all files) between old and new dataset")
    sum = df.sum(axis=0)
    index_diff = len(df1.columns)
    for i in range(1,index_diff):
        print(f"loss diff for {df1.columns[i]}: {sum[i]-sum[i+index_diff-1]}")


if __name__ == "__main__":
    main()
