# Financial-Time-Series
Time series models (Transformer,  Foundation Models) for Financial data 

## Dataset

### Currency Exchange Rate

Representative rates of US dollar for the period August 01, 2014 - August 01, 2024.  
Collected from the [IMF rates database](https://www.imf.org/external/np/fin/ert/GUI/Pages/CountryDataBase.aspx).
These rates, normally quoted as currency units per U.S. dollar, are reported daily to the Fund by the issuing central bank. (The IMF does not maintain exchange rates on weekends and some holidays.) The collected data covers the following currencies:

1. Australian Dollar (AUD)
2. Candian Dollar (CAD)
3. Chinese yuan(CNY)
4. Euro(EUR)
5. Indian rupee(INR)
6. Japanese yen(JPY)
7. U.K. pound(GBP)

Converted to csv using the following

```python
df = pd.read_csv('./data/Exchange_Rate_Report.tsv', sep='\t')
df.drop(['Unnamed: 0', 'Unnamed: 9'], axis=1, inplace=True)
df.fillna(method='ffill').fillna(method='bfill').to_csv(
    './data/Exchange_Rate_Report.csv', 
    sep=',', index=False
)
```

### Stock Market

Daily stock prices (Close, Open, High, Low) and volumes for each stock for upto 10 years from NASDAQ database. 

1. [S&P 500 (SPX)](https://www.nasdaq.com/market-activity/index/spx/historical?page=1&rows_per_page=10&timeline=y10)
2. [Microsoft Corporation (MSFT)](https://www.nasdaq.com/market-activity/stocks/msft/historical)
3. [Apple Inc. (AAPL)](https://www.nasdaq.com/market-activity/stocks/aapl/historical?page=1&rows_per_page=10&timeline=y10)

### Commodity

1. [Natural Gas](https://www.nasdaq.com/market-activity/commodities/ng-nmx/historical?page=1&rows_per_page=10&timeline=y10) 
2. [Crude oil](https://www.nasdaq.com/market-activity/commodities/cl-nmx/historical?page=1&rows_per_page=10&timeline=y10)
3. [Gold](https://www.nasdaq.com/market-activity/commodities/gc-cmx/historical?page=1&rows_per_page=10&timeline=y10)

# Models

Time Series models implemented using the [Time Series Library](https://github.com/thuml/Time-Series-Library)
1. DLinear
2. MICN
3. SegRNN
4. iTransformer
5. TimeMixer
6. PatchTST
7. TimesNet

Time Series LLM models
1. [One Fits All (OFA) : Power General Time Series Analysis by Pretrained LM](https://arxiv.org/abs/2302.11939) (NeurIPS 2023 Spotlight)
2. [CALF - Aligning LLMs for Time Series Forecasting via Cross-modal Fine-Tuning.](https://arxiv.org/abs/2403.07300) 
3. [Time-LLM: Time Series Forecasting by Reprogramming Large Language Models](https://arxiv.org/pdf/2310.01728)