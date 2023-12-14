import os
import torch
from matplotlib import pyplot as plt
from numpy import mean
from torch.optim.lr_scheduler import ExponentialLR

from Classificator.LSTM import LSTM
from Train.Load_Data import get_dataloader, multi_data
from Classificator.Generator import Generator

device = 'cuda' if torch.cuda.is_available() else 'cpu'

title = "None"


def files_name(file_dir):
    res = list()
    for root, dirs, files in os.walk(file_dir, topdown=False):
        res.append(files)
    return res[0]


class Discriminate_Train(object):

    def __init__(self, model):
        super(Discriminate_Train, self).__init__()
        self.criterion = torch.nn.CrossEntropyLoss()  # 损失函数
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 优化器
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.scheduler = ExponentialLR(self.optimizer, gamma=0.9)
        self.train_loss = []
        self.accuracy = []

    def train(self, model, data_loader):

        sub_train_loss = []
        cnt = 0
        for data, target in data_loader:
            model.train()
            data = data.to(self.device)  # 部署到device
            target = target.to(self.device)
            self.optimizer.zero_grad()  # 梯度置零

            out = model(data)  # 模型训练

            loss = self.criterion(out, target)  # 计算损失
            loss.backward()  # 反向传播
            self.optimizer.step()  # 参数更新
            sub_train_loss.append(loss.item())
            cnt += 1
            if cnt >= 1000:
                break

        loss_per_epoch = mean(sub_train_loss)
        self.train_loss.append(mean(loss_per_epoch))
        print("*******************************************************************************")
        print("*******************************************************************************")
        print("loss : {}\n".format(loss_per_epoch))



    def test(self, test_model, test_loader_set):
        test_model.eval()

        correct = 0.0
        total = 0.0

        cnt = 0

        for packet, labels in test_loader_set:
            packet = packet.to(self.device)
            labels = labels.to(self.device)
            out = test_model(packet)
            predict_res = torch.max(out.data, 1)[1]
            # print("labels.size() = {}".format(labels.size()))
            total += labels.numel()

            if torch.cuda.is_available():
                correct += (predict_res.cuda(0) == labels.cuda(0)).sum()
            else:
                correct += (predict_res == labels).sum()

            cnt += 1
            if cnt >= 1000:
                break

        accuracy = correct / total * 100

        print("*******************************************************************************")
        print("*******************************************************************************")
        print("Accuracy : {}\n".format(accuracy))
        self.accuracy.append(accuracy.to('cpu'))



    def show(self, epoch):
        plt.plot(self.train_loss)
        plt.xlabel('number of epoch')
        plt.ylabel('loss' + "___test")
        plt.title(epoch)
        plt.show()

        plt.plot(self.accuracy)
        plt.xlabel('number of epoch')
        plt.ylabel('accuracy')
        plt.title(epoch)
        plt.show()

        self.train_loss.clear()
        self.accuracy.clear()



    def save(self, model, model_save_path):
        torch.save(model.state_dict(), model_save_path)


if __name__ == '__main__':

    # n-fold交叉验证：

    files = files_name("../DataSet/Train_Data/")

    data_list = multi_data("../DataSet/Train_Data/")

    model = LSTM(11, 128, 2, 14)
    model = model.to(device)
    dicriminate_train = Discriminate_Train(model)

    data_len = len(data_list)

    for epoch in range(50):

        for index in range(data_len):

            print("当前测试集为：{}".format(files[index]))

            for i in range(data_len):

                if i != index:

                    dicriminate_train.train(model, data_list[i])

            dicriminate_train.test(model, data_list[index])

            print("============================================================================")
            print("****************************************************************************")
            print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
            print("Total Accuracy：{}".format(mean(dicriminate_train.accuracy)))

        dicriminate_train.show(str(epoch))

    dicriminate_train.save(model, "../Saved_Model/20220709/" + "n-fold.pkl")