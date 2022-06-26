import pandas as pd
from matplotlib import pyplot as plt

class ChartView():
    def __init__(self, csv_path):
        data = pd.read_csv(csv_path)
        self.data = pd.DataFrame({
            "datetime": data["datetime_jst"],
            "close": data["close"]
        })
        self.data = self.data.set_index("datetime")
        self.data.plot()
        plt.show()



if __name__ == "__main__":
    view = ChartView(csv_path="dataset/7days_1min/3107.csv")


