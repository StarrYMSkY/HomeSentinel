import os
import torch
from matplotlib import pyplot as plt
from pandas import read_csv, DataFrame
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from Model.Generator import Generator
from Train.Load_Data import get_all_loader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
noise_size = 11


def files_name(file_dir):
    res = list()
    for root, dirs, files in os.walk(file_dir, topdown=False):
        res.append(files)
    return res[0]


def data_plot(file):
    data = read_csv("../DataSet/Train_Data/" + file)
    values = data.values[:, :-1]
    label = data.values[:, -1:]
    data = DataFrame(values)

    n_components = 2
    tsne = TSNE(n_components=n_components)
    tsne_res = tsne.fit_transform(data)

    x = []
    y = []

    for item in tsne_res:
        x.append(item[0])
        y.append(item[1])
    return x, y, label.tolist()

if __name__ == '__main__':
    # data_plot("360drop_lan_wlan0.csv")

    file_list = files_name("../DataSet/Train_Data/")

    X_List = [[] for i in range(14)]
    Y_List = [[] for i in range(14)]
    Adv_X_List = [[] for i in range(14)]
    Adv_Y_List = [[] for i in range(14)]

    for file in file_list:
        # model_name = "../Saved_Model/20220802/Model_" + file[:-4] + ".pkl"
        file_name = "../DataSet/Train_Data/" + file

        print("----------------------------------------------------------------------------------------")
        print("----------------------------------------------------------------------------------------")
        # print(model_name)
        print(file_name)

        generator = Generator(128, 256, 2, 11)
        generator.load_state_dict(torch.load("../Saved_Model/Generator.pkl", map_location='cpu'))

        data_loader, maxn_0, maxn_9 = get_all_loader(file_name)
        generator = generator.to(device)

        datalist = []
        labellist = []

        for data, target in data_loader:
            generator.eval()
            data = data.to(device)  # 部署到device
            target = target.to(device)  # 部署到device
            label = target.view(target.size()[0], 1)
            # 生成服从正态分布的随机噪音数据
            rand_noise = (torch.randn(data.size()[0], 128) / 10.0)
            rand_noise = rand_noise.to(device)

            rand_noise = rand_noise.to(device)

            adv_data = generator(rand_noise)

            adv_data = adv_data + data
            # adv_data = torch.cat([adv_data + data, target.view(-1, 1)], 1)

            labellist.append(target)
            datalist.append(adv_data)


        dataframe = datalist[0]
        labelframe = labellist[0]
        for index in range(1, len(datalist)):
            dataframe = torch.cat([dataframe, datalist[index]], 0)
            labelframe = torch.cat([labelframe, labellist[index]], 0)

        dataframe = dataframe.detach().cpu().numpy()
        labelframe = labelframe.detach().cpu().numpy()
        data_len = len(dataframe)

        for index in range(data_len):
            dataframe[index][0] = int(dataframe[index][0] * maxn_0)
            dataframe[index][9] = int(dataframe[index][9] * maxn_9)

        n_components = 2
        tsne = TSNE(n_components=n_components)
        tsne_res = tsne.fit_transform(dataframe)

        x = []
        y = []

        for item in tsne_res:
            x.append(item[0])
            y.append(item[1])

        xx, yy, label = data_plot(file)

        for index in range(data_len):
            X_List[labelframe[index]].append(xx[index])
            Y_List[labelframe[index]].append(yy[index])
            Adv_X_List[int(label[index])].append(x[index])
            Adv_Y_List[int(label[index])].append(y[index])



    for index in range(14):

        plt.scatter(X_List, Y_List, c='blue', s=1)
        plt.scatter(Adv_X_List, Adv_Y_List, c='red', s=1)
        plt.title(index)
        plt.savefig('../Image/' + file[:-4] + ".png")
        plt.show()

