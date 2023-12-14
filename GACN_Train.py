import os
import torch
from torch import Tensor
from torch.autograd import Variable

from Adv_Example.Model.Discriminator import Discriminator
from Adv_Example.Model.Generator import Generator
from Classificator.LSTM import LSTM
from Utils.Load_Data import get_all_loader, get_dataloader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
noise_size = 11

def files_name(file_dir):
    res = list()
    for root, dirs, files in os.walk(file_dir, topdown=False):
        res.append(files)
    return res[0]


class GAN_Train(object):

    def __init__(self, generator, discriminator):
        super(GAN_Train, self).__init__()
        self.criterion = torch.nn.MSELoss()  # 损失函数
        self.optimizer_gen = torch.optim.Adam(generator.parameters(), lr=0.0001)  # 优化器
        self.optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr=0.0001)  # 优化器
        self.num_epochs = 50  # 循环次数

    def train(self, generator, discriminator, classificator, data_loader):

        optimizer_cls = torch.optim.Adam(classificator.parameters(), lr=0.0001)

        for epoch in range(self.num_epochs):
            iter = 0
            G_loss = 0.0
            D_loss = 0.0
            for data, target in data_loader:
                data = data.to(device)
                target = target.to(device)
                # label = target.view(target.size()[0], 1)
                # ————————————————————————————————————————————————————————
                # 训练生成器
                generator.train()
                self.optimizer_gen.zero_grad()  # 梯度置零

                rand_noise = torch.randn(data.size()[0], 128) / 10.0

                rand_noise = rand_noise.to(device)

                # rand_noise = rand_noise + data

                # rand_noise = torch.cat([rand_noise, label], dim=1)

                # 生成器生成假数据
                generated_noise = generator(rand_noise)

                valid = Variable(Tensor(data.shape[0], 1).fill_(1.0), requires_grad=False).to(device)
                fake = Variable(Tensor(data.shape[0], 1).fill_(0.0), requires_grad=False).to(device)

                discriminated_label = discriminator(generated_noise + data)

                categoory = classificator(generated_noise + data)

                optimizer_cls.zero_grad()  # 梯度置零

                c_loss = torch.nn.CrossEntropyLoss()(categoory, target)  # 计算损失

                g_loss = self.criterion(discriminated_label, valid)  # 计算损失

                g_loss = (g_loss - c_loss) * 0.5

                g_loss.backward()  # 反向传播
                self.optimizer_gen.step()  # 参数更新

                discriminator.train()
                # ——————————————————————————————————————————————————————————
                self.optimizer_dis.zero_grad()

                # Measure discriminator's ability to classify real from generated samples
                real_loss = self.criterion(discriminator(data), valid)
                fake_loss = self.criterion(discriminator(data + generated_noise.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2

                d_loss.backward()
                self.optimizer_dis.step()

                G_loss += g_loss
                D_loss += d_loss
                iter += 1

            # if epoch > 30:
            for param_group in self.optimizer_gen.param_groups:
                param_group["lr"] *= 0.95  # 优化器参数更新

            for param_group in self.optimizer_dis.param_groups:
                param_group["lr"] *= 0.95  # 优化器参数更新

            print("-----------------------------------------------------------------------")
            print("epoch : {}, G_Loss : {}, D_Loss : {}".format(epoch, G_loss/iter, D_loss/iter))



    def save(self, model, model_save_path):
        torch.save(model.state_dict(), model_save_path)


if __name__ == '__main__':
    generator = Generator(128, 128, 2, 5)
    discriminator = Discriminator(5, 128, 2, 1)

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    gan_train = GAN_Train(generator, discriminator)

    file_list = files_name("DataSet/Train_Data2/")

    for file in file_list:
        model_name = "Saved_Model/20220928_Model_" + file[:-4] + ".pkl"
        file_name = "DataSet/Train_Data/" + file

        classificator = LSTM(5, 128, 2, 29)
        classificator.load_state_dict(torch.load("Saved_Model/Model2/20220928_Model_" + file[:-4] + ".pkl"))
        classificator = classificator.to(device)
        print("----------------------------------------------------------------------------------------")
        print("----------------------------------------------------------------------------------------")
        print(file_name)
        data_loader, test_loader = get_dataloader("DataSet/Train_Data2/" + file)
        gan_train.train(generator, discriminator, classificator, data_loader)


    gan_train.save(generator, "Saved_Model/Model2/20220930-Generator.pkl")
    gan_train.save(discriminator, "Saved_Model/Model2/20220930-Discriminator.pkl")
