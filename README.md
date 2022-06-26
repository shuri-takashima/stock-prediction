# stock-prediction
This Program Code is stock prediction with AI!

```
Env Anaconda
python 3.6.13
torch 1.8.0
GPU  NVIDIA:Geforce RTX 3060Ti
```

if you use code, Check option.py!
```python
python main.py
```
steps_dataloader in option.py with True  is a stock separeted csv row!  
steps_dataloader in option.py with False  isn't sck separeted csv row! 

you want to use dataset, Check download_dataset/sample.py or download.py.
sample.py get a stock.
donwload.py get all stocck.
if you don't get purpose stock, you will change dataloader.


### Train loss (RMSE)
|NIKKEI| 1day for 10years | 1min for 7days |
| ---- | ---- | ---- |
| LSTM | 0.073 | 0.063 |
| GRU | 0.069 | 0.044 |
| Liner | 0.379 | 0.56|

### Test loss (RMSE)
|NIKKEI| 1day for 10years | 1min for 7days |
| ---- | ---- | ---- |
| LSTM | 0.450 | 0.527 |
| GRU | 0.491 | 0.531 |
| Liner | 0.870 | 0.887|

Ummm!  
It's difficult to predict stock.   
I guess The stock is complicated, There are a lot of factor.   
1day for 10 years with LSTM and GRU is 50% or less.  
So this model is overlearning, it's should learn more data.  

