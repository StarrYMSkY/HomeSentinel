import torch
import torch.nn as nn


class Discriminator(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_dim):
        super(Discriminator, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, output_dim)
        self.sig = nn.Sigmoid()
        self.device = 'cuda'

    def forward(self, x):
        x = x.view(len(x), 1, -1)
        x.to(self.device)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(self.device)
        # 初始化cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(self.device)
        # 分离隐藏状态，以免梯度爆炸
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        # 只需要最后一层隐层的状态
        out = out[:, -1, :]
        out = self.drop(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.sig(out)
        return out