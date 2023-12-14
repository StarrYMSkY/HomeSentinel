import os

from pandas import read_csv


def files_name(file_dir):
    res = list()
    for root, dirs, files in os.walk(file_dir, topdown=False):
        res.append(files)
    return res[0]

if __name__ == '__main__':

    path = "DataSet/Train_Data/"
    files_list = files_name(path)
    count = [0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0]
    for item in files_list:
        data = read_csv(path + item)
        values = data.values.tolist()
        # print(values)
        for it in values:
            # print(it[-1])
            count[int(it[-1])] += 1
    print(count)
