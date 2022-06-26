import torch
from torch import nn
from torch.utils.data import DataLoader
from dataloader import MyDataset
from option import Option

from networks.liner_net import Liner_Net
from networks.lstm_net import LSTM_Net
from networks.gru_net import GRU_Net

class Train():
    def __init__(self):
        self.option = Option()
        if self.option.model_name == "Liner":
            self.model = Liner_Net(
            ).to(self.option.device)

        if self.option.model_name=="LSTM":
            self.model = LSTM_Net(
            ).to(self.option.device)
        if self.option.model_name =="GRU":
            self.model = GRU_Net(
            ).to(self.option.device)

        self.model.train()
        self.model.load_state_dict(torch.load(self.option.model_path))

        dataset = MyDataset()
        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.option.batch_size,
            shuffle=True,
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.option.lr)

    def MSE_Loss(self, y_pred, y):
        if (y.dim() == 3):
            y = torch.reshape(y,(y.shape[0], y.shape[1]))
        if (y_pred.dim() == 3):
            y_pred = torch.reshape(y_pred, (y_pred.shape[0], y_pred.shape[1]))
        criterion = nn.MSELoss(reduction='mean')
        loss = criterion(y_pred, y)
        return loss

    def train(self):
        loss_list=[]
        for epoch in range(self.option.epochs+1):
            for i, data in enumerate(self.dataloader):
                x = data[0].to(self.option.device)
                y = data[1].to(self.option.device)

                if self.option.model_name == "Liner":
                    x = torch.reshape(x, (x.shape[0], x.shape[1]))
                    y = torch.reshape(y, (y.shape[0], y.shape[1]))

                y_pred = self.model(x)
                loss = self.MSE_Loss(y_pred, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_list.append(loss.item())
                # print("i:", i, loss.item())
            print("Epoch:", epoch, "loss:", loss.item(), "y_pred mean:",torch.mean(y_pred).item())
            self.save_model(model=self.model, epoch=epoch)
        print(torch.mean(torch.Tensor(loss_list)).item())


    def save_model(self, model, epoch):
        torch.save(model.state_dict(), "{}/{}.pth".format(self.option.save_dir, epoch))


if __name__ == "__main__":
    T = Train()
    T.train()



