import numpy
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from option import Option
from dataloader import MyDataset
from networks.gru_net import GRU_Net
from networks.liner_net import Liner_Net
from networks.lstm_net import LSTM_Net


class Evaluation():
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

        self.model.state_dict(torch.load(self.option.model_path))
        print("model load successfully")
        self.model.eval()

        dataset = MyDataset()
        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.option.batch_size,
            #shuffle=True,
        )

    def RMSE_loss(self, y_pred, y):
        if (y.dim() == 3):
            y = torch.reshape(y, (y.shape[0], y.shape[1]))
        if (y_pred.dim() == 3):
            y_pred = torch.reshape(y_pred, (y_pred.shape[0], y_pred.shape[1]))
        criterion = nn.MSELoss()
        loss = criterion(y_pred,y)
        # loss = torch.sqrt(criterion(y_pred, y))
        return loss

    def eval(self):
        loss_list=[]
        for i, data in enumerate(self.dataloader):
            x = data[0].to(self.option.device)
            y = data[1].to(self.option.device)
            y_pred = self.model(x)
            loss = self.RMSE_loss(y_pred, y)
            loss_list.append(loss)

            # print("loss", loss.item())

        print("loss mean", torch.mean(torch.Tensor(loss_list)).item())


    def drew_fig(self):
        for i in self.dataloader:
            x = i[0].to(self.option.device)
            y = i[1].to(self.option.device)
            y_pred = self.model(x)

            x = x.cpu().detach().numpy()
            y =y.cpu().detach().numpy()
            y_pred = y_pred.cpu().detach().numpy()
            x = np.reshape(x, (x.shape[0], x.shape[1]))
            y = np.reshape(y, (y.shape[0], y.shape[1]))
            y_pred = np.reshape(y_pred, (y_pred.shape[0], y_pred.shape[1]))

            print("正解",y[0], "予測",y_pred[0])
            real = np.append(x, y)
            pred = np.append(x,y_pred)
            plt.plot(real[0], color="blue")

if __name__ == "__main__":
    Eval= Evaluation()
    Eval.eval()
    Eval.drew_fig()




