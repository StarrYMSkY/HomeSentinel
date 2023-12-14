import os
import torch
from matplotlib import pyplot as plt

from Classificator.LSTM import LSTM
from Utils.Load_Data import get_dataloader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

title = "None"


def files_name(file_dir):
    res = list()
    for root, dirs, files in os.walk(file_dir, topdown=False):
        res.append(files)
    return res[0]


class Classificator_Train(object):

    def __init__(self, model):
        super(Classificator_Train, self).__init__()
        self.criterion = torch.nn.CrossEntropyLoss()  # 损失函数
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 优化器
        self.num_epochs = 50  # 循环次数
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.train_loss = []
        self.accuracy = []


    def train(self, model, data_loader, test_loader):

        train_loss = []

        for epoch in range(self.num_epochs):
            model.train()
            total_loss = 0.0
            cnt = 0

            # if epoch > 30:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] *= 0.97  # 优化器参数更新

            for data, target in data_loader:

                data = data.to(self.device)  # 部署到device
                target = target.to(self.device)
                self.optimizer.zero_grad()  # 梯度置零

                out = model(data)  # 模型训练

                loss = self.criterion(out, target)  # 计算损失
                train_loss.append(loss.item())  # 累计损失
                loss.backward()  # 反向传播
                self.optimizer.step()  # 参数更新

                total_loss += loss.item()
                cnt += 1

            self.train_loss.append(total_loss / cnt)
            print("------------------------------------------------------------------------------------")
            print("Epoch : {}, loss : {}\n".format(epoch, total_loss / cnt))

        self.test(model, test_loader)


    def test(self, model, test_loader):

        model.eval()

        correct = 0.0
        total = 0.0

        for packet, labels in test_loader:
            packet = packet.to(self.device)
            labels = labels.to(self.device)
            out = model(packet)
            predict = torch.max(out.data, 1)[1]
            # print("labels.size() = {}".format(labels.size()))
            total += labels.numel()

            if torch.cuda.is_available():
                correct += (predict.cuda(0) == labels.cuda(0)).sum()
            else:
                correct += (predict == labels).sum()

        # print("correct = {}, total = {}".format(correct, total))
        accuracy = correct / total * 100

        print("*******************************************************************************")
        print("*******************************************************************************")
        # print("correct : {}, total : {}".format(correct, total))
        print("Accuracy : {}\n".format(accuracy))

        self.accuracy.append(accuracy.cpu())


    def show(self, title):
        plt.plot(self.train_loss)
        plt.xlabel('number of epoch')
        plt.ylabel('loss' + "___test")
        plt.title(title + "__loss")
        plt.show()

        plt.plot(self.accuracy)
        plt.xlabel('number of epoch')
        plt.ylabel('accuracy')
        plt.title(title + "__accuracy")
        plt.show()

        self.accuracy.clear()
        self.train_loss.clear()


    def save(self, model, model_save_path):
        torch.save(model.state_dict(), model_save_path)

def single_train():
    path = "DataSet/Train_Data2/"
    files_list = files_name(path)

    #Train_Data1
    # input_dim = 11
    # output_dim = 13

    #Train_Data2
    # input_dim = 11
    # output_dim = 28

    # Train_Data3
    # input_dim = 11
    # output_dim = 8
    matrix = [[0, 0, 0, 0, 0, 0, 0, 0] for i in range(8)]

    for file_name in files_list:

        print(path + file_name)

        model = LSTM(input_dim, 128, 2, output_dim)
        model = model.to(device)
        classificator_train = Classificator_Train(model)

        train_loader, test_loader = get_dataloader(path + file_name)

        classificator_train.train(model, train_loader, test_loader)

        print("********************************************************")
        print("********************************************************")
        print(file_name, "  处理完毕！")
        print("********************************************************")
        print("********************************************************\n\n")

        classificator_train.save(model, "Saved_Model/Model2/20220928_Model_" + file_name[:-4] + ".pkl")


def train_multi():

    path = "DataSet/Train_Data2/"
    files_list = files_name(path)

    #Train_Data1
    # input_dim = 5
    # output_dim = 13

    #Train_Data2
    input_dim = 5
    output_dim = 29

    model = LSTM(input_dim, 128, 2, output_dim)
    model = model.to(device)
    classificator_train = Classificator_Train(model)

    for file_name in files_list:

        print(path + file_name)

        train_loader, test_loader = get_dataloader(path + file_name)

        classificator_train.train(model, train_loader, test_loader)

        print("********************************************************")
        print("********************************************************")
        print(file_name, "  处理完毕！")
        print("********************************************************")
        print("********************************************************\n\n")

    classificator_train.save(model, "Saved_Model/Model2/20220912_Total_Model.pkl")


if __name__ == '__main__':

    print(torch.cuda.is_available())

    # single_train()

    # train_multi()








