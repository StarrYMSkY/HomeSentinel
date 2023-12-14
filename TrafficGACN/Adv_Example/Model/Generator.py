import torch
import torch.nn as nn


class Generator(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_dim):
        super(Generator, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hidden_size * 2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, x):
        x = x.view(len(x), 1, -1)
        x.to(self.device)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).requires_grad_().to(self.device)
        # 初始化cell state
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).requires_grad_().to(self.device)
        # 分离隐藏状态，以免梯度爆炸
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        # 只需要最后一层隐层的状态
        out = out[:, -1, :]
        out = self.sigmoid(out)
        out = self.drop(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out