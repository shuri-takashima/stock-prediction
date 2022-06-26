import os
import glob
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

from option import Option

class MyDataset(Dataset):
    def __init__(self):
        self.option = Option()
        self.csv_list = glob.glob(self.option.csv_dir)
        self.input_feature = self.option.input_feature
        self.pd_list = self.csvlist_to_pdlist()
        print(len(self.pd_list))

    def csvlist_to_pdlist(self):
        pd_list=[]
        for csv in self.csv_list:
            old_df = pd.read_csv(csv)
            new_df = pd.DataFrame(old_df[self.input_feature])
            if self.option.pct_change:
                new_df = new_df.pct_change()
            new_df = new_df.fillna(0)
            new_np = new_df.to_numpy()
            if self.option.steps_dataloader:
                for index in range(len(new_np) - self.option.input_seq_dim):
                    pd_list.append(new_np[index:index+self.option.input_seq_dim+1]) #予測する値の+1
            else:
                pd_list.append(new_np)
        return pd_list

    def __getitem__(self, item):
        all = torch.Tensor(self.pd_list[item]).type(torch.Tensor)
        mmscaler = MinMaxScaler(feature_range=(-1,1), copy=True)
        mmscaler.fit(all)
        all = mmscaler.transform(all)
        all = torch.from_numpy(all).type(torch.Tensor)
        x = all[:self.option.input_seq_dim]
        y = all[self.option.input_seq_dim:self.option.input_seq_dim+self.option.output_seq_dim]
        return (x, y)

    def __len__(self):
        return len(self.pd_list)


if __name__ == "__main__":
    D = MyDataset()
    item=0
    x, y=D.__getitem__(item)
    print(x.shape, y.shape)