import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(
              os.path.dirname(__file__),
              os.pardir))
sys.path.append(PROJECT_ROOT)
import utils

"""
Run tests: 'pytest test_rnn_model.py'
"""

def test_handle_names_are_wrong():
    data = {
        "lha_y": [2],
        "rha_y": [3],
        "lsh_y": [5],
        "rsh_y": [1],
    }
    df = pd.DataFrame(data)
    assert utils.handle_names_are_wrong(df)

def test_fix_axis_values():
    data = {
        "lha_y": [2],
        "rha_y": [3],
        "lsh_y": [5],
        "rsh_y": [1],
    }
    df = pd.DataFrame(data)
    true_data = {
        "lha_y": [3],
        "rha_y": [2],
        "lsh_y": [5],
        "rsh_y": [1],
    }
    true_df = pd.DataFrame(true_data)
    new_df = utils.fix_handle_names(df)
    assert new_df.equals(true_df)

def test_fix_axis_values_timesteps():
    data = {
        "lha_y": [3,-1, 88],
        "rha_y": [1, 2,195],
        "rsh_y": [5, 2,-21],
    }
    df = pd.DataFrame(data)
    true_data = {
        "lha_y": [1, 2,195],
        "rha_y": [3,-1, 88],
        "rsh_y": [5, 2,-21],
    }
    true_df = pd.DataFrame(true_data)
    new_df = utils.fix_handle_names(df)
    assert new_df.equals(true_df)

def test_data_to_angle():
    data1 = [1,1,1]
    data2 = [1,1,1]
    data3 = [1,1,1]

    data4 = [0,0,0]
    data5 = [0,1,1]
    data6 = [0,0,1]

    data7 = [2,2,2]
    data8 = [2,2,2]
    data9 = [2,0,1]

    data = {
        'p1_x' : data1,
        'p1_y' : data2,
        'p1_z' : data3,

        'p2_x' : data4,
        'p2_y' : data5,
        'p2_z' : data6,

        'p3_x' : data7,
        'p3_y' : data8,
        'p3_z' : data9,
    }

    df = pd.DataFrame(data)
    angle_lst = utils.data_to_angle(df,[['p1','p2','p3']])
    solution = np.radians(np.array([[180],[90],[135]]))
    assert(angle_lst == np.array(solution)).all()

def test_get_angle_array():
    data = np.array([
        [[1,1,1],[1,1,1],[1,1,1]],
        [[0,0,0],[0,1,0],[0,1,1]],
        [[2,2,2],[2,2,0],[2,2,1]]
    ])

    correct_data = np.radians(np.array([180,90,135]))

    assert(utils.get_angle_array(data[0],data[1],data[2]).all() == correct_data.all())

def test_angle_between_two_line():
    data = np.array([
        [[1,1,1],
        [0,0,0],
        [2,2,2]],
        [[1,1,1],
        [0,1,0],
        [2,2,0]],
        [[1,1,1],
        [0,1,1],
        [2,2,1]],
    ])

    correct_data = np.radians(np.array([180,90,135]))

    for i in range(len(data)):
        assert(utils.angle_between_two_line(data[i][0],data[i][1],data[i][2]) == correct_data[i])

def test_split_data_ones():
    data = np.array([
        [1,2,5,6],
        [1,7,3,9],
        [1,4,7,8],
        [1,7,5,5],
        [5,9,6,8],
        [8,8,1,0]])
    correct_data = np.array([
        [[1,2,5,6],[1,7,3,9]],
        [[1,7,3,9],[1,4,7,8]],
        [[1,4,7,8],[1,7,5,5]],
        [[1,7,5,5],[5,9,6,8]],
        [[5,9,6,8],[8,8,1,0]]])
    correct_labels = np.ones(correct_data.shape[0])
    new_data,new_labels = utils.split_data(data,2,1,1)
    assert (new_data == correct_data).all()
    assert (new_labels == correct_labels).all()

def test_split_data_zeros():
    data = np.array([
        [1,2,5,6],
        [1,7,3,9],
        [1,4,7,8],
        [1,7,5,5],
        [5,9,6,8],
        [8,8,1,0]])
    correct_data = np.array([
        [[1,2,5,6],[1,7,3,9],[1,4,7,8]],
        [[1,7,3,9],[1,4,7,8],[1,7,5,5]],
        [[1,4,7,8],[1,7,5,5],[5,9,6,8]],
        [[1,7,5,5],[5,9,6,8],[8,8,1,0]]])
    correct_labels = np.zeros(correct_data.shape[0])
    new_data,new_labels = utils.split_data(data,3,2,0)
    assert (new_data == correct_data).all()
    assert (new_labels == correct_labels).all()

def test_split_data_zero_padding():
    data = np.array([
        [1,2,5,6],
        [1,7,3,9],
        [1,4,7,8],
        [1,7,5,5],
        [5,9,6,8],
        [8,8,1,0]])
    correct_data = np.array([
        [[1,2,5,6],[1,7,3,9],[1,4,7,8]],
        [[1,4,7,8],[1,7,5,5],[5,9,6,8]],
        [[5,9,6,8],[8,8,1,0],[0,0,0,0]]])
    correct_labels = np.zeros(correct_data.shape[0])
    new_data,new_labels = utils.split_data(data,3,1,0)
    assert (new_data == correct_data).all()
    assert (new_labels == correct_labels).all()

def test_data_to_xyz_features():
    assert (utils.to_xyz_features(np.array([[1,2,3,4,5,6]])) == np.array([[[1,2,3],[4,5,6]]])).all()

def test_normalize():
    assert (utils.normalize(np.array([1,2,3])) == np.array([0,0.5,1])).all()
    assert (utils.normalize(np.array([[1,2],[3,2]])) == np.array([[0,0.5],[1,0.5]])).all()
    np.testing.assert_almost_equal(utils.normalize(np.array([3,15,8])),np.array([0,1,0.41667]),decimal=5)

def test_filename_to_id():
    assert utils.filename_to_id("src/id7_phml_c4_mocap.csv") == 7
    assert utils.filename_to_id("id10_phml_c4_mocap.csv") == 10
    assert utils.filename_to_id("src/id2_phml_c2_mocap.csv") == 2

def test_filename_to_label():
    assert utils.filename_to_label("id7_phml_c4_mocap.csv") == 1
    assert utils.filename_to_label("src/id2_phml_c2_mocap.csv") == 1
    assert utils.filename_to_label("src/id13_phml_c1_mocap.csv") == 0
