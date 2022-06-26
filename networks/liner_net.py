import sys
sys.path.append("../")
import torch
from torch import nn
from option import Option

class Liner_Net(nn.Module):
    def __init__(self):
        super(Liner_Net, self).__init__()
        self.option= Option()
        self.l1 = self.layer(self.option.input_seq_dim, self.option.hidden_dim)
        self.l2 = self.layer(self.option.hidden_dim, self.option.hidden_dim)
        self.l3 = self.layer(self.option.hidden_dim, self.option.hidden_dim)
        self.l4 = self.layer(self.option.hidden_dim, self.option.output_seq_dim)

    def layer(self, in_feature, out_feature):
        layer = nn.Sequential(
            nn.Linear(in_feature, out_feature),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        return layer


    def forward(self, x):
        y = self.l1(x)
        y = self.l2(y)
        y = self.l3(y)
        y = self.l4(y)
        return y

if __name__ == "__main__":
    x = torch.rand(size=(3,30))
    m = Liner_Net()
    y = m(x)
    print(y.shape)