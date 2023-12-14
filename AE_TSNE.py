import os
import time
import torch
from matplotlib import pyplot as plt
from pandas import read_csv
from sklearn.decomposition import PCA, KernelPCA, LatentDirichletAllocation
from sklearn.manifold import TSNE

from Model.Generator import Generator
from Utils.Load_Data import get_all_loader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
noise_size = 11
seed = 39
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
        # model_name = "Saved_Model/20220802/Model_" + file[:-4] + ".pkl"
        file_name = "DataSet/Train_Data/" + file
        if file != '360drop_lan_wlan0.csv':
            continue

        print("----------------------------------------------------------------------------------------")
        print("----------------------------------------------------------------------------------------")
        # print(model_name)
        print(file_name)

        generator = Generator(128, 128, 2, 11)
        # generator.load_state_dict(torch.load("Saved_Model/Model1/20220918-Generator.pkl", map_location='cpu'))
        generator.load_state_dict(torch.load("Saved_Model/Model1/Generator-20220918.pkl"))

        data_loader, maxn_0, maxn_9 = get_all_loader(file_name)
        data_csv = read_csv("DataSet/Train_Data/" + file)
        generator = generator.to(device)

        datalist = []
        labellist = []

        for data, target in data_loader:
            generator.eval()
            data = data.to(device)  # 部署到device
            target = target.to(device)  # 部署到device
            label = target.view(target.size()[0], 1)
            # 生成服从正态分布的随机噪音数据
            time0 = time.time()
            rand_noise = (torch.randn(data.size()[0], 128) / 10)
            print(rand_noise.shape)
            rand_noise = rand_noise.to(device)

            adv_data = generator(rand_noise)
            # adv_data = rand_noise

            adv_data = torch.cat([adv_data + data, target.view(-1, 1)], 1)

            datalist.append(adv_data)

            print(time.time() - time0)




        # dataframe = datalist[0]

        # for index in range(1, len(datalist)):
        #     dataframe = torch.cat([dataframe, datalist[index]], 0)

        # dataframe = dataframe.detach().cpu().numpy()

        # data_len = len(dataframe)

        # for index in range(data_len):
        #     dataframe[index][0] = int(dataframe[index][0] * maxn_0)
        #     dataframe[index][9] = int(dataframe[index][9] * maxn_9)


        # tsne = TSNE(n_components=n_components, random_state=seed)
        # res = tsne.fit_transform(dataframe)

        # x = []
        # y = []

        # for item in res:
        #     x.append(item[0])
        #     y.append(item[1])

        # xx, yy = data_plot(file)

        # plt.figure(dpi=300)
        # plt.scatter(xx, yy, c='blue', marker=".", s=1)
        # plt.scatter(x, y, c='red', marker=".", s=1)
        # # plt.title(file)
        # plt.legend(["Dummy IoT traffic", "Real IoT traffic"], loc='upper right')
        # plt.savefig('Image/GASN_and_real-' + file[:-4] + ".png")
        # plt.show()
        


