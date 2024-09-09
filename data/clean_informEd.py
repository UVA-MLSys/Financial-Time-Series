import pandas as pd
# https://informedstates.org/state-financial-aid-dataset-download
df = pd.read_excel('data/InformEd_States_statefaid_dataset.xlsx', sheet_name='data')
print(df.head(3))

df.rename(columns={'stateid': 'GROUP_ID', 'year': 'Date'}, inplace=True)
df = df[['GROUP_ID', 'Date', 'need_amt', 'need_number']]
df.to_csv('data/Financial_Aid_State.csv', index=False)