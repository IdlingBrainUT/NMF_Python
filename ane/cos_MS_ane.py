# calc cos similarity and Matching Score from the result of NMF (.csv)
# This code was tuned for the experiments using anesthetized/control mice.
import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import os
from matplotlib import pyplot as plt

""" Input ( Start ) """

dirname = "basis"

""" Input ( End ) """

flist = os.listdir(dirname)
namelist = []
for f in flist:
    if f.split(".")[-1] == "csv":
        if f.split("_day")[0] not in namelist:
            namelist.append(f.split("_day")[0])

if "dot_img" not in os.listdir():
    os.mkdir("dot_img")
if "dot_csv" not in os.listdir():
    os.mkdir("dot_csv")
if "MS_img" not in os.listdir():
    os.mkdir("MS_img")
if "MS_csv" not in os.listdir():
    os.mkdir("MS_csv")

""" Color map settings """
cdict1 = {'red':   ((0.0, 0.0, 0.0),
                   (0.5999, 1.0, 1.0),
                   (1.0, 1.0, 1.0)),

         'green': ((0.0, 0.0, 0.0),
                   (0.5999, 1.0, 0.0),
                   (1.0, 1.0, 1.0)),

         'blue':  ((0.0, 1.0, 1.0),
                   (0.5999, 1.0, 0.0),
                   (1.0, 0.0, 0.0))
        }

# colormap like MatLab
cm_data = [[0.2081, 0.1663, 0.5292], [0.2116238095, 0.1897809524, 0.5776761905], 
           [0.212252381, 0.2137714286, 0.6269714286], [0.2081, 0.2386, 0.6770857143], 
           [0.1959047619, 0.2644571429, 0.7279], [0.1707285714, 0.2919380952, 0.779247619],
           [0.1252714286, 0.3242428571, 0.8302714286], [0.0591333333, 0.3598333333, 0.8683333333],
           [0.0116952381, 0.3875095238, 0.8819571429], [0.0059571429, 0.4086142857, 0.8828428571], 
           [0.0165142857, 0.4266, 0.8786333333], [0.032852381, 0.4430428571, 0.8719571429],
           [0.0498142857, 0.4585714286, 0.8640571429], [0.0629333333, 0.4736904762, 0.8554380952],
           [0.0722666667, 0.4886666667, 0.8467], [0.0779428571, 0.5039857143, 0.8383714286], 
           [0.079347619, 0.5200238095, 0.8311809524], [0.0749428571, 0.5375428571, 0.8262714286],
           [0.0640571429, 0.5569857143, 0.8239571429], [0.0487714286, 0.5772238095, 0.8228285714],
           [0.0343428571, 0.5965809524, 0.819852381], [0.0265, 0.6137, 0.8135],
           [0.0238904762, 0.6286619048, 0.8037619048], [0.0230904762, 0.6417857143, 0.7912666667], 
           [0.0227714286, 0.6534857143, 0.7767571429], [0.0266619048, 0.6641952381, 0.7607190476],
           [0.0383714286, 0.6742714286, 0.743552381], [0.0589714286, 0.6837571429, 0.7253857143], 
           [0.0843, 0.6928333333, 0.7061666667], [0.1132952381, 0.7015, 0.6858571429], 
           [0.1452714286, 0.7097571429, 0.6646285714], [0.1801333333, 0.7176571429, 0.6424333333],
           [0.2178285714, 0.7250428571, 0.6192619048], [0.2586428571, 0.7317142857, 0.5954285714],
           [0.3021714286, 0.7376047619, 0.5711857143], [0.3481666667, 0.7424333333, 0.5472666667], 
           [0.3952571429, 0.7459, 0.5244428571], [0.4420095238, 0.7480809524, 0.5033142857],
           [0.4871238095, 0.7490619048, 0.4839761905], [0.5300285714, 0.7491142857, 0.4661142857],
           [0.5708571429, 0.7485190476, 0.4493904762], [0.609852381, 0.7473142857, 0.4336857143], 
           [0.6473, 0.7456, 0.4188], [0.6834190476, 0.7434761905, 0.4044333333], 
           [0.7184095238, 0.7411333333, 0.3904761905], [0.7524857143, 0.7384, 0.3768142857],
           [0.7858428571, 0.7355666667, 0.3632714286], [0.8185047619, 0.7327333333, 0.3497904762], 
           [0.8506571429, 0.7299, 0.3360285714], [0.8824333333, 0.7274333333, 0.3217], 
           [0.9139333333, 0.7257857143, 0.3062761905], [0.9449571429, 0.7261142857, 0.2886428571],
           [0.9738952381, 0.7313952381, 0.266647619], [0.9937714286, 0.7454571429, 0.240347619],
           [0.9990428571, 0.7653142857, 0.2164142857], [0.9955333333, 0.7860571429, 0.196652381], 
           [0.988, 0.8066, 0.1793666667], [0.9788571429, 0.8271428571, 0.1633142857], 
           [0.9697, 0.8481380952, 0.147452381], [0.9625857143, 0.8705142857, 0.1309], 
           [0.9588714286, 0.8949, 0.1132428571], [0.9598238095, 0.9218333333, 0.0948380952],
           [0.9661, 0.9514428571, 0.0755333333], [0.9763, 0.9831, 0.0538]
        ]
mycm = LinearSegmentedColormap('mycm', cdict1)
mycm2 = LinearSegmentedColormap.from_list('mycm2', cm_data)

def readfile(filename):
    df = pd.read_csv(dirname + "/" + filename, header=None)
    return df.values

d1_in_d2 = []
d2_in_d1 = []
for name in namelist:
    print(name)    
    lst = [f for f in flist if name in f]
    lst = sorted(lst)
    numlist = [f.split(".")[0].split("_")[1] + "_" + f.split(".")[0].split("_")[2] for f in lst]
    print(lst)
    datalist = [readfile(f) for f in lst]
    MS = np.ones((len(lst),len(lst))) * -1
    for i, f1 in enumerate(lst):
        for j, f2 in enumerate(lst[i+1:]):
            """ calc cos similarity """
            dots = np.dot(datalist[i], datalist[j+i+1].T)

            norm1 = np.linalg.norm(datalist[i], ord=2, axis=1)
            norm2 = np.linalg.norm(datalist[j+i+1], ord=2, axis=1)

            dots /= norm1.reshape([norm1.shape[0], 1])
            dots /= norm2

            """ save to png (cos sim.) """
            fig, ax = plt.subplots()
            heatmap = ax.pcolor(dots, cmap=mycm, vmin=0, vmax=1)
            fig.colorbar(heatmap)
            ax.set_xlabel(numlist[j+i+1])
            ax.set_ylabel(numlist[i])
            plt.title(name + "_" + numlist[i] + "_vs_" + numlist[j+i+1])
            plt.savefig("dot_img/" + name + "_" + numlist[i] + "_vs_" + numlist[j+i+1] + ".png")
            plt.clf()
            plt.close()

            """ save to csv (cos sim.) """
            df = pd.DataFrame(dots, index=np.arange(dots.shape[0]), columns=np.arange(dots.shape[1]))
            df.to_csv("dot_csv/" + name + "_" + numlist[i] + "_vs_" + numlist[j+i+1] + ".csv")

            """ calc Matching Score """
            MS[i][j+i+1] = np.sum(np.amax(dots, axis=0)>=0.6) / datalist[j+i+1].shape[0]
            MS[j+i+1][i] = np.sum(np.amax(dots, axis=1)>=0.6) / datalist[i].shape[0]
            d2_in_d1.append(MS[i][j+i+1])
            d1_in_d2.append(MS[j+i+1][i])

    """ save to png (MS) """
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(MS, cmap=mycm2, vmin=0, vmax=1)
    heatmap.cmap.set_over("k")
    heatmap.cmap.set_under("k")
    plt.colorbar(heatmap)
    plt.title("MS for " + name)
    plt.xlabel("ref")
    plt.ylabel("in")
    plt.savefig("MS_img/" + name + "_MS.png")
    plt.clf()
    plt.close()

    """ save to csv (MS) """
    df = pd.DataFrame(MS, index=np.arange(MS.shape[0]), columns=np.arange(MS.shape[1]))
    df.to_csv("MS_csv/" + name + "_MS.csv")

""" create bar plot (MS) """
d1_in_d2_ane = np.array([d1_in_d2[0], d1_in_d2[1], d1_in_d2[3]]) # set index according to datasets
d1_in_d2_con = np.array([d1_in_d2[2], d1_in_d2[4]])
d2_in_d1_ane = np.array([d2_in_d1[0], d2_in_d1[1], d2_in_d1[3]])
d2_in_d1_con = np.array([d2_in_d1[2], d2_in_d1[4]])
d1_in_d2_ane_sem = d1_in_d2_ane.std(ddof=1) / np.sqrt(3)
d1_in_d2_con_sem = d1_in_d2_con.std(ddof=1) / np.sqrt(2)
d2_in_d1_ane_sem = d2_in_d1_ane.std(ddof=1) / np.sqrt(3)
d2_in_d1_con_sem = d2_in_d1_con.std(ddof=1) / np.sqrt(2)
lab = ["ane", "con"]
plt.bar(lab, [d1_in_d2_ane.mean(), d1_in_d2_con.mean()], yerr=[d1_in_d2_ane_sem, d1_in_d2_con_sem])
plt.title("pattern-day1 in pattern-day2")
plt.ylabel("MS")
plt.savefig("d1_in_d2.png")
plt.close()
plt.bar(lab, [d2_in_d1_ane.mean(), d2_in_d1_con.mean()], yerr=[d2_in_d1_ane_sem, d2_in_d1_con_sem])
plt.title("pattern-day2 in pattern-day1")
plt.ylabel("MS")
plt.savefig("d2_in_d1.png")
plt.close()