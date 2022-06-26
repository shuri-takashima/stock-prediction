import sys
sys.path.append("../")
import torch
from torch import nn
from option import Option

class GRU_Net(nn.Module):
    def __init__(self):
        super(GRU_Net, self).__init__()
        self.option = Option()

        self.gru = nn.GRU(
            self.option.input_data_dim,
            self.option.hidden_dim,
            self.option.num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(self.option.hidden_dim, self.option.output_data_dim)


    def forward(self, x):
        h0 = torch.zeros(self.option.num_layers, x.size(0), self.option.hidden_dim, device=self.option.device).requires_grad_()
        out, (hn) = self.gru(x, (h0.detach()))
        out = self.fc(out[:, -1, :])
        return out

if __name__ == "__main__":
    G = GRU_Net().to("cuda:0")
    x = torch.randn(size=(2,30,1)).to("cuda:0")
    y = G(x)
    print(y.shape)