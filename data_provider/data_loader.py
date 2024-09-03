import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')

class Dataset_Custom(Dataset):
    def __init__(
        self, args, flag='train', time_column='Date'
    ):
        # size [seq_len, label_len, pred_len]
        self.args = args
    
        self.seq_len = args.seq_len
        self.label_len = args.label_len
        self.pred_len = args.pred_len

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = args.features
        self.target = args.target
        self.scale = not args.no_scale
        self.timeenc = 0 if args.embed != 'timeF' else 1
        self.freq = args.freq

        self.root_path = args.root_path
        self.data_path = args.data_path
        self.time_column = time_column
        self.scaler = {}
        self.__read_data__()

    def __read_data__(self):
        
        df_raw = pd.read_csv(
            os.path.join(self.root_path, self.data_path)
        )
        df_raw.rename(columns={self.time_column: 'date'}, inplace=True)
        df_raw['date'] = pd.to_datetime(df_raw.date)
        df_raw.sort_values(by=['date'], inplace=True)
        
        # some files have $ in values
        object_columns = df_raw.select_dtypes(include=['object']).columns
        df_raw[object_columns] = df_raw[object_columns].apply(
            lambda x: x.str.replace('$', '')
        )
        df_raw = df_raw.fillna(method='ffill').fillna(method='bfill')

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove('date')
        
        if self.features != 'M':
            cols.remove(self.target)
            df_raw = df_raw[['date'] + cols + [self.target]]
        else:
            df_raw = df_raw[['date'] + cols]
            
        num_train = int(len(df_raw) * 0.8)
        num_test = int(len(df_raw) * 0.1)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # features
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        # if self.scale:
        #     train_data = df_data[border1s[0]:border2s[0]]
            
        #     self.scaler = StandardScaler()
        #     self.scaler.fit(train_data.values)
        #     data = self.scaler.transform(
        #         df_data[border1:border2].values
        #     )
        # else:
        #     data = df_data[border1:border2].values
        
        self.index = df_stamp = df_raw[['date']][border1:border2].reset_index(drop=True)
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = self.data_y = df_data[border1:border2].values
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        
        if self.scale:
            scaler = StandardScaler().fit(seq_x)

            seq_x = scaler.transform(seq_x) 
            seq_y = scaler.transform(seq_y)
            first_date = self.index.loc[s_begin, 'date']
            self.scaler[first_date] = scaler
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, scaler, data):
        if not self.scale: return data
        if scaler is None:
            print('Error ! The scaler is None.')
            return data
        
        if len(data.shape)>2:
            prev_shape = data.shape
            data = data.reshape(-1, data.shape[-1])
            data = scaler.inverse_transform(data)
            return data.reshape(prev_shape)
        
        return scaler.inverse_transform(data)