import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJECT_ROOT = os.path.abspath(os.path.join(
              os.path.dirname(__file__),
              os.pardir))
sys.path.append(PROJECT_ROOT)

file_path = os.path.dirname(os.path.realpath(__file__))

def plot_mean_std(plt,x,acc_mean,acc_std,label,coords=True):
    acc_min = np.array(acc_mean)-np.array(acc_std)
    acc_max = np.array(acc_mean)+np.array(acc_std)

    m = 'o' if coords else 'x'

    plt.plot(x,acc_min,color='grey',linestyle='dashed')
    plt.plot(x,acc_mean,marker=m,label=label)
    plt.plot(x,acc_max,color='grey',linestyle='dashed')

    plt.title("Accuracy for feature combinations")
    plt.savefig("plot_acc_features")

def calculate_mean(values):
    return [
        np.mean(values[i]) for i in range(values.shape[0])
    ]

def calculate_median(values):
    return [
        np.median(values[i]) for i in range(values.shape[0])
    ]

def calculate_std(values):
    return [
        np.std(values[i]) for i in range(values.shape[0])
    ]

def plot_violin_plot(acc_all,pos,color):
    acc_all = list(acc_all)
    violin_parts = plt.violinplot(acc_all,positions=pos,showmeans=True)#,quantiles=[.25,.75])

    # Color all the violin statistics marks:
    for partname in ('cbars','cmins','cmaxes','cmeans'):#,'cmedians'):
        vp = violin_parts[partname]
        vp.set_edgecolor('black')
        vp.set_linewidth(1)

    # Color the violin body
    for vp in violin_parts['bodies']:
        vp.set_facecolor(color)
        #vp.set_edgecolor(rrred)
        vp.set_linewidth(1)
        vp.set_alpha(0.5)

def acc_df_to_array(df,n):
    acc_values_list = []
    for i in range(n):
        acc_values_list.append(df.loc[df['feature_nr'] == (i+1)]['accuracy'].to_numpy())
    acc_values = np.array(acc_values_list)
    return acc_values

def epochs_to_array(df,n,use_angles):
    df = df.loc[df['feature_type'] == bool(use_angles)]

    epochs_list = []
    for i in range(n):
        epochs_list.append(df.loc[df['feature_nr'] == (i+1)]['epoch'].to_numpy())
    epochs = np.array(epochs_list)
    return epochs

def print_epoch_scores(epochs_coords,epochs_angles):
    epocs_mean_coords = calculate_mean(epochs_coords)
    epocs_median_coords = calculate_median(epochs_coords)
    epocs_std_coords = calculate_std(epochs_coords)

    epocs_mean_angles = calculate_mean(epochs_angles)
    epocs_median_angles = calculate_median(epochs_angles)
    epocs_std_angles = calculate_std(epochs_angles)

    print("\nEPOCHS COORDS")
    print("Mean epochs coords",np.round(epocs_mean_coords,4),sep="\n")
    print("St.d epochs coords",np.round(epocs_std_coords,4),sep="\n")
    print("Median epochs coords",np.round(epocs_median_coords,4),sep="\n")

    print("\nEPOCHS ANGLES")
    print("Mean epochs angles",np.round(epocs_mean_angles,4),sep="\n")
    print("St.d epochs angles",np.round(epocs_std_angles,4),sep="\n")
    print("Median epochs angles",np.round(epocs_median_angles,4),sep="\n")

def print_acc_scores(coords,angles,type):
    acc_mean_coords = calculate_mean(coords)
    acc_median_coords = calculate_median(coords)
    acc_std_coords = calculate_std(coords)

    acc_mean_angles = calculate_mean(angles)
    acc_median_angles = calculate_median(angles)
    acc_std_angles = calculate_std(angles)

    print("\n"+type+" COORDS")
    print("Mean acc coords",np.round(acc_mean_coords,4),sep="\n")
    print("St.d acc coords",np.round(acc_std_coords,4),sep="\n")
    print("Median acc coords",np.round(acc_median_coords,4),sep="\n")

    print("\n"+type+" ANGLES")
    print("Mean acc angles",np.round(acc_mean_angles,4),sep="\n")
    print("St.d acc angles",np.round(acc_std_angles,4),sep="\n")
    print("Median acc angles",np.round(acc_median_angles,4),sep="\n")


def main():
    # Epochs
    epochs_file = os.path.join(file_path,"..","out/metrics","gru_epochs.csv")
    epochs_df = pd.read_csv(epochs_file)

    epochs_coords = epochs_to_array(epochs_df,5,False)
    epochs_angles = epochs_to_array(epochs_df,3,True)

    print_epoch_scores(epochs_coords,epochs_angles)
    print("\n")

    # Accuracies
    angles_test_file = os.path.join(file_path,"..","out/metrics","gru_angles_test.csv")
    coords_test_file = os.path.join(file_path,"..","out/metrics","gru_coords_test.csv")

    angles_test2_file = os.path.join(file_path,"..","out/metrics","gru_2_angles_test.csv")
    coords_test2_file = os.path.join(file_path,"..","out/metrics","gru_2_coords_test.csv")

    angles_val_file = os.path.join(file_path,"..","out/metrics","gru_angles_val.csv")
    coords_val_file = os.path.join(file_path,"..","out/metrics","gru_coords_val.csv")

    angles_test_df = pd.read_csv(angles_test_file)
    coords_test_df = pd.read_csv(coords_test_file)

    angles_test2_df = pd.read_csv(angles_test2_file)
    coords_test2_df = pd.read_csv(coords_test2_file)

    angles_val_df = pd.read_csv(angles_val_file)
    coords_val_df = pd.read_csv(coords_val_file)

    # Load test values
    angles_test = acc_df_to_array(angles_test_df,3)
    coords_test = acc_df_to_array(coords_test_df,5)

    angles_test2 = acc_df_to_array(angles_test2_df,3)
    coords_test2 = acc_df_to_array(coords_test2_df,5)

    angles_val = acc_df_to_array(angles_val_df,3)
    coords_val = acc_df_to_array(coords_val_df,5)

    print_acc_scores(coords_val,angles_val,"VAL")
    print_acc_scores(coords_test,angles_test,"TEST")
    print_acc_scores(coords_test2,angles_test2,"TEST2")

    titlesize = 18
    fontsize = 14

    datasets = [[coords_val,angles_val],[coords_test,angles_test],[coords_test2,angles_test2]]
    acc_name = ["Val","Test","Test2"]
    for i in range(len(datasets)):
        filename = "violinplot_features_"+acc_name[i].lower()+"_acc"
        title = acc_name[i]+" accuracy for feature combinations"

        dataset = datasets[i]

        plt.figure()

        x = range(dataset[0].shape[0]+dataset[1].shape[0])
        n_coords = dataset[0].shape[0]
        n_angles = dataset[1].shape[0]


        ticks_coords = ["ShoHip","ShoSea","ShoFro","HanFro","HipFro"]
        ticks_angles = ["\u03B1","\u03B2","\u03B3"]
        #ticks_coords = ["F"+str(i+1) for i in range(n_coords)]
        #ticks_angles = ["F"+str(i+1) for i in range(n_angles)]
        ticks_names = ticks_coords + ticks_angles

        plt.xticks(x,ticks_names)
        pos = [range(n_coords),range(n_coords,n_angles+n_coords)]

        colors = ['red','blue']
        coords_patch = mpatches.Patch(color=colors[0])
        angles_patch = mpatches.Patch(color=colors[1])
        fake_handles = [coords_patch,angles_patch]

        plt.axhline(y = 0.5, color = 'grey', linestyle = ':')

        for i in range(len(dataset)):
            plot_violin_plot(dataset[i],pos[i],colors[i])

        plt.xlabel("Feature combination number",fontsize=fontsize)
        plt.ylabel("Accuracy",fontsize=fontsize)
        plt.title(title,fontsize=titlesize)
        plt.legend(fake_handles, ["coords","angles"])
        plt.savefig(filename)
        plt.show()

if __name__ == "__main__":
    main()
