import sys
sys.path.append('../')
import torch
from torch import nn
from option import Option

class LSTM_Net(nn.Module):
    def __init__(self):
        super(LSTM_Net, self).__init__()
        self.option= Option()

        self.lstm = nn.LSTM(
            self.option.input_data_dim,
            self.option.hidden_dim,
            self.option.num_layers,
            batch_first=True,
            dropout=self.option.dropout
        )
        self.fc = nn.Linear(self.option.hidden_dim, self.option.output_data_dim)

    def forward(self, x):
        h0 = torch.zeros(self.option.num_layers, x.size(0), self.option.hidden_dim, device=self.option.device).requires_grad_()
        c0 = torch.zeros(self.option.num_layers, x.size(0), self.option.hidden_dim, device=self.option.device).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out

if __name__ == "__main__":
    x = torch.randn(size=(2,30,1))
    L = LSTM_Net()
    y = L.forward(x=x)
    print(y.shape)