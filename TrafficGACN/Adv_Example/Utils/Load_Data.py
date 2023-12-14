import os

import numpy.random
import torch
from pandas import read_csv
from torch.utils.data import TensorDataset, DataLoader

batch_size = 128


def file_name(file_dir):
    res = list()
    for root, dirs, files in os.walk(file_dir, topdown=False):
        res.append(files)
    return res[0]



def load_data(file):

    data = read_csv(file)

    # print(data.head())

    data = data.values

    maxn_0 = data.max(axis=0)[0]
    maxn_9 = data.max(axis=0)[9]

    data_len = len(data)

    for index in range(data_len):
        data[index][0] = float(data[index][0]) / maxn_0
        data[index][9] = float(data[index][9]) / maxn_9

    # print(data_len)

    # print(data)

    train_index = int(0.8 * data_len)

    numpy.random.seed(39)
    numpy.random.shuffle(data)

    train_x, train_y = data[:train_index, :-1], data[:train_index, -1:]
    test_x, test_y = data[train_index:, :-1], data[train_index:, -1:]
    train_y = train_y.flatten()
    test_y = test_y.flatten()

    return train_x, train_y, test_x, test_y



def get_dataloader(file_name):
    train_x, train_y, test_x, test_y = load_data(file_name)
    # print("---------------------------------------------")
    # for item in train_y:
    #     print(item)
    train_x_tensor = torch.from_numpy(train_x).float()
    train_y_tensor = torch.from_numpy(train_y).long()
    test_x_tensor = torch.from_numpy(test_x).float()
    test_y_tensor = torch.from_numpy(test_y).long()

    train_x_tensor = torch.nn.functional.normalize(train_x_tensor)
    # train_y_tensor = torch.nn.functional.normalize(train_y_tensor)
    test_x_tensor = torch.nn.functional.normalize(test_x_tensor)
    # test_y_tensor = torch.nn.functional.normalize(test_y_tensor)
    # print("test_loader:{},{},{}".format(train_x_tensor.size(), val_x_tensor.size(), test_x_tensor.size()))
    train_dataset = TensorDataset(train_x_tensor, train_y_tensor)
    test_dataset = TensorDataset(test_x_tensor, test_y_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=False, drop_last=True)

    return train_loader, test_loader



def load_all_data(file):

    data = read_csv(file)

    # print(data.head())

    data = data.values

    maxn_0 = data.max(axis=0)[0]
    maxn_9 = data.max(axis=0)[9]

    data_len = len(data)

    for index in range(data_len):
        data[index][0] = float(data[index][0]) / maxn_0
        data[index][9] = float(data[index][9]) / maxn_9

    numpy.random.seed(39)
    numpy.random.shuffle(data)

    data_x, data_y = data[:, :-1], data[:, -1:]
    data_y = data_y.flatten()

    return data_x, data_y, maxn_0, maxn_9



def get_all_loader(file_name):
    data_x, data_y, maxn_0, maxn_9 = load_all_data(file_name)

    data_x_tensor = torch.from_numpy(data_x).float()
    data_x_tensor = torch.nn.functional.normalize(data_x_tensor)
    data_y_tensor = torch.from_numpy(data_y).long()

    dataset = TensorDataset(data_x_tensor, data_y_tensor)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=False, drop_last=True)

    return data_loader, maxn_0, maxn_9


# if __name__ == '__main__':
#     res = load_n_data("../DataSet/Train_Data/360drop_lan_wlan0.csv")
#     print(len(res))