import torch
import torch.nn as nn


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, bidirectional=True, dropout=0.5)
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.fc2 = nn.Linear(64, output_dim)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, x):
        x = x.view(len(x), 1, -1)
        h0 = torch.zeros(self.num_layers * 2, x.size(0),
                         self.hidden_size).requires_grad_().to(self.device)
        # 初始化cell state
        c0 = torch.zeros(self.num_layers * 2, x.size(0),
                         self.hidden_size).requires_grad_().to(self.device)
        # 分离隐藏状态，以免梯度爆炸
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        # 只需要最后一层隐层的状态
        out = out[:, -1, :]
        out = self.sigmoid(out)
        out = self.drop(out)
        out = self.fc1(out)
        out = self.fc2(out)

        return out
