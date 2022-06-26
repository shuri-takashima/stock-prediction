# stock-prediction

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


### loss
|NIKKEI| 1day for 10years | 1min for 7days |
| ---- | ---- | ---- |
| LSTM | 0.445 | 0.527 |
| GRU | 0.590 | 0.599 |
| Liner | 0.4996 | 0.887|

