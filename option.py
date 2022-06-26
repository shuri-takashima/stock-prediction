import os

class Option():
    def __init__(self):
        self.train=True
        self.device ="cuda:0"
        self.model_name = "GRU" #Liner or LSTM or GRU
        self.csv_dir = "./dataset/NIKKEI/10years_1day" #glob表記で記入
        self.save_dir="models/GRU/NIKKEI/10years_1day"
        os.makedirs(self.save_dir, exist_ok=True)
        self.model_path="models/GRU/7days_1min/100.pth"
        self.input_feature = "close"

        self.epochs = 100
        self.batch_size=32
        self.lr =0.01
        self.input_seq_dim = 30
        self.input_data_dim=1#入力するシーケンス数　入力する期間
        self.output_seq_dim=1
        self.output_data_dim=1#出力するシーケンス数　予測する期間
        self.hidden_dim= 30
        self.num_layers=3
        self.dropout=0.2
        self.steps_dataloader = True #ex)一株に１minごとに刻んでデータセットを作成する。
        self.pct_change=False #Recomend False


