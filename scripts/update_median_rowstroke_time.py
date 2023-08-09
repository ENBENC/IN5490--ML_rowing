import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(
              os.path.dirname(__file__),
              os.pardir))
sys.path.append(PROJECT_ROOT)

from utils import all_files, filename_to_separator, timedata, timestep_size

def calculate_median_stroke_length(file):
    """
    This function will take a rowing datafrime, and find median time of a row stroke.

    Args:
        df (pandas DataFrame): rowing datafram

    Return:
        median (float): median time of row strokes in seconds
    """
    def same_direction(dirc1, dirc2):
        """
        Find that 2 direction input is in same direction.
        Direction is define by positive float og negative float.

        EX:
            If direc1 is 1.90 and direc2 is 3.32, then return true.

        Args:
            direc1 (float): direction 1
            direc2 (float): direction 2

        Returns:
            (boolean) : direc1 and direc2 is same direction
        """
        if(dirc1 < 0 and dirc2 < 0):
            return True
        elif(dirc1 > 0 and dirc2 > 0):
            return True
        else:
            return False

    def diff_positions(pos1, pos2):
        """
            Take two position and find difference between them,
            and return absolutt value of the difference.

        Args:
            pos1 (float): position 1
            pos2 (float): position 2

        Returns:
            (float): abs of difference of tow positions
        """
        return abs(pos1-pos2)

    def one_stroke_lst(half_strok_lst):
        """
        Take a list of time, where each time represents time of half strok.
        This function take 2 half strok time into 1 full strok time.

        Args:
            half_strok_lst (list) : list of half stroke times

        Returns:
            full_stroke_lst (list) : list of full stroke times
        """
        if(len(half_strok_lst) % 2 != 0):
            #Remove last element, for half_strok_lst that is odd length
            half_strok_lst = half_strok_lst[:-1]

        full_stroke_lst = []
        i1 = 0
        i2 = 1
        for i in range(int(len(half_strok_lst)/2)):
            full_stroke_lst.append(half_strok_lst[i1]+half_strok_lst[i2])
            i1 = i2+1
            i2 = i1+1

        return full_stroke_lst

    def find_start_stop(file):
        if 'src/' in file:
            file = file.split("src/")[1]
        row = timedata.loc[timedata['Filename'] == file]
        start = int((row['Start']+10)*timestep_size)
        stop = int((row['Stop']-10)*timestep_size)
        return start, stop

    data = pd.read_csv(file,sep=filename_to_separator(file))
    df = pd.DataFrame(data)

    start, stop = find_start_stop(file)
    df = df[start:stop]

    col = df['lha_x'].to_numpy()

    #Define time step, where we check for each 0.25 secound
    #The corresponding index to 0.25s is 60
    time_step = 0.25
    index_step = 60

    #Index to keep index and time during iteration
    current_i = 0
    current_t = 0

    #Variable keep start time and end time of a strok
    start_t = 0
    end_t = 0

    direction = col[index_step] - col[0]
    lst = []

    while(current_i + index_step < col.shape[0]):

        next_diection = col[current_i + index_step] - col[current_i]
        #count a strok when direction is not same, and position difference is greater than 1mm
        if(not(same_direction(direction,next_diection)) and
        diff_positions(col[current_i], col[current_i + index_step]) > 1):
            end_t = current_t
            current_i += index_step
            direction = next_diection

            lst.append(end_t-start_t)

            start_t = current_t
            current_t += time_step


        else:
            current_t += time_step
            current_i += index_step
            direction = next_diection;

    lst = one_stroke_lst(lst)
    median = np.median(lst)
    return median

def main():
    file_path = os.path.dirname(os.path.realpath(__file__))

    with open(os.path.join(file_path,'../median_strokes.txt'), 'w') as output_file:

        for file in all_files:
            rowing_file = os.path.join(file_path,"..",file)
            median_stroke = calculate_median_stroke_length(rowing_file)
            if median_stroke >= 3:
                median_stroke = 2.75 #To avoid too much of a difference between strokes
            filename = file
            if "/" in filename:
                filename = filename.split("/")[1]
            output_file.write(filename+","+str(median_stroke)+"\n")

    output_file.close()

if __name__ == "__main__":
    main()
