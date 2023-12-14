import torch
import torch.nn as nn



device = 'cuda' if torch.cuda.is_available() else 'cpu'



class LSTMDiscriminator(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_dim):
        super(LSTMDiscriminator, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(2 * hidden_size, output_dim)

    def forward(self, x):
        x = x.view(len(x), 1, -1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(device)
        # 初始化cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(device)
        # 分离隐藏状态，以免梯度爆炸
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        # 只需要最后一层隐层的状态
        out = out[:, -1, :]
        out = self.drop(out)
        out = self.fc(out)
        return out