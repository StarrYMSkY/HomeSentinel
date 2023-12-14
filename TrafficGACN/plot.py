import os
import random
import torch
from matplotlib import pyplot as plt
from pandas import read_csv
from sklearn.decomposition import PCA, KernelPCA, LatentDirichletAllocation
from sklearn.manifold import TSNE

from Model.Generator import Generator
from Utils.Load_Data import get_all_loader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
noise_size = 11
seed = None
n_components = 2

def files_name(file_dir):
    res = list()
    for root, dirs, files in os.walk(file_dir, topdown=False):
        res.append(files)
    return res[0]


def data_plot(file):
    data = read_csv("DataSet/Train_Data/" + file)

    n_components = 2
    tsne = TSNE(n_components=n_components, random_state=seed)
    res = tsne.fit_transform(data)

    x = []
    y = []

    for item in res:
        x.append(item[0])
        y.append(item[1])
    return x, y

if __name__ == '__main__':
    # data_plot()

    file_list = files_name("DataSet/Train_Data/")

    for file in file_list:
        xx, yy = data_plot(file)

        x = [random.uniform(-75, 75) for i in range(len(xx))]
        y = [random.uniform(-75, 75) for i in range(len(yy))]

        plt.figure(dpi=300)
        plt.scatter(xx, yy, c='blue', marker="^", s=1)
        plt.scatter(x, y, c='red', marker="^", s=1)
        # plt.title(file)
        plt.legend(["Dummy IoT traffic", "Real IoT traffic"], loc='upper right')
        plt.savefig('Image/noise_and_real' + file[:-4] + ".png")
        plt.show()
        


