import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from utils.timefeatures import time_features
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

class MultiTimeSeriesMultiTarget(Dataset):
    def __init__(
        self, args, flag='train', time_col='year'
    ):
        # size [seq_len, label_len, pred_len]
        # info

        self.seq_len = args.seq_len
        self.label_len = args.label_len
        self.pred_len = args.pred_len
        self.flag = flag
        
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        
        self.features = args.features
        self.target = [
            'need_amt', 'need_number', 'need_avg', 
            'merit_amt', 'merit_number', 'merit_avg', 
            'combo_amt', 'combo_number', 'combo_avg'
        ] 
        self.scale = not args.no_scale
        self.split = (0.8, 0.1, 0.1)
        self.dry_run = args.dry_run
        
        self.args = args
        self.ranges = None
        
        self.group_id = args.group_id
        self.time_col = time_col
        self.time_steps = self.seq_len + self.pred_len
        self.scaler = {}
        self.__read_data__()
        
    def split_data(self, df_raw):
        time_col = self.time_col
        min_year = df_raw[time_col].min()
        
        self.min_year = min_year
        df_raw[time_col] -= min_year
        
        # split the data into train, val, test by time column
        years = sorted(df_raw[time_col].unique())
        train_split, val_split, test_split = self.split

        train_start = 0
        train_end = int(len(years) * train_split)
        val_end = int(len(years) * (train_split + val_split))
        test_end = len(years)

        if self.set_type == 0:
            print('Minimum year ', min_year)
            print(f"Split years:\n \
                Train: {[y+min_year for y in years[train_start: train_end]]}\n \
                Validation: {[y+min_year for y in years[train_end: val_end]]}\n \
                Test: {[y+min_year for y in years[val_end: test_end]]}\n")
        
        # Set borders
        border1s = [0, train_end - self.seq_len, val_end - self.seq_len]
        border2s = [train_end, val_end, test_end]
        
        border1 = years[border1s[self.set_type]]
        border2 = years[border2s[self.set_type]-1]
        
        if self.set_type == 0 and self.args.percent < 100:
            percent = self.args.percent
            length = border2 - self.seq_len - self.pred_len - border1
            if length < 100:
                print(f'Warning: Length {length} is less than 100. Chosing by groups.')
                groups = df_raw[self.group_id].unique()
                choice_size = len(groups) * percent//100
                chosen_groups = np.random.choice(groups, choice_size, replace=False)
                df_raw = df_raw[df_raw[self.group_id].isin(chosen_groups)]
                print(f'Selected {len(chosen_groups)} groups among {len(groups)}.')
            else:
                border1 = length * (100-percent)//100 + border1
                
        self.border1, self.border2 = border1, border2
        
        # filter out data
        df_data = df_raw[
            (df_raw[time_col]>=self.border1) & (df_raw[time_col]<=self.border2)
        ].copy().reset_index(drop=True)
        return df_data
        
    def __read_data__(self):
        # read and clean dataset
        df = pd.read_csv(os.path.join(self.args.root_path,self.args.data_path))
        time_steps = self.time_steps
        
        df.sort_values(by=self.time_col, inplace=True)
        
        df = self.split_data(df)
        df, selected_columns = self.scale_data(df)
        ranges, data, data_stamp = [], [], []
        
        for identifier, df in tqdm(
            df.groupby(self.group_id), desc='Preparing data',
            disable=self.args.disable_progress
        ):
            num_entries = len(df)
            if num_entries < time_steps: continue
            
            for i in range(num_entries - time_steps + 1):
                sliced = df.iloc[i:i + time_steps]
                data.append(sliced[selected_columns])
    
                data_stamp.append(sliced[[self.time_col]].values)
                ranges.append((identifier, i))
            if self.dry_run: break

        self.data = np.array(data)
        self.data_stamp = np.array(data_stamp)      
        self.ranges = ranges
        
    @property    
    def index(self):
        indices = []
        for identifier, start_idx in self.ranges:
            indices.append(
                (identifier, start_idx + self.border1 + self.seq_len + self.min_year)
            )
            
        return pd.DataFrame(
            indices, columns=[self.group_id, self.time_col]
        )
            
    # TODO: add group scaling
    def scale_data(self, df_raw):
        group_id, time_col = self.group_id, self.time_col
        target = self.target
        print('Scaling data.')
        # get input features
        input_cols = [
            col for col in df_raw.columns \
                if col not in [group_id, time_col] + target
        ]
        # not all input columns will not be selected for training
        #Important: targets need to be at the end
        if self.features == 'S': 
            selected_columns = target
            # input feature categories by type
            categorical_features = []
        else: 
            selected_columns = [group_id] + input_cols + target
            categorical_features = [group_id] + input_cols
        
        # selected_columns = [col for col in selected_columns if col not in categorical_features]
        self.selected_columns = selected_columns
        
        # only print one for training data, no need for repetition
        if self.set_type == 0:
            print(f'Categoricals or ID: {categorical_features}.')
            print(f'Time column {time_col}, target {target}.')
        
        # not using categorical features for now
        # scaling must be done for categorical features
        if len(categorical_features) > 0:
            # Must scale categorical or object features
            cat_encoder = OrdinalEncoder(dtype=int)
            # fitted on whole dataset to handle unknown labels
            cat_encoder.fit(df_raw[categorical_features])
            df_raw.loc[:, categorical_features] = cat_encoder.transform(
                df_raw[categorical_features]
            )
            
        return df_raw, selected_columns
        
    def __getitem__(self, index):
        s_end = self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data[index][:s_end]
        seq_y = self.data[index][r_begin:r_end]
        
        seq_x_mark = self.data_stamp[index][:s_end]
        seq_y_mark = self.data_stamp[index][r_begin:r_end]
        
        if self.scale:
            n_targets = len(self.target)
            scaler = StandardScaler().fit(seq_x[:,-n_targets:])
            seq_x[:, -n_targets:] = scaler.transform(seq_x[:, -n_targets:])
            seq_y[:, -n_targets:] = scaler.transform(seq_y[:, -n_targets:])
            
            self.scaler[index] = scaler
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data) # - self.seq_len - self.pred_len + 1
    
    def inverse_transform(self, scaler, data):
        if not self.scale: return data
        if scaler is None:
            print('Error ! The scaler is None.')
            return data
        
        if len(data.shape)==1:
            data = scaler.inverse_transform(data.reshape(-1, len(self.target)))
            return data.flatten()
        elif len(data.shape)>2:
            prev_shape = data.shape
            data = data.reshape(-1, data.shape[-1])
            data = scaler.inverse_transform(data)
            return data.reshape(prev_shape)
        
        return scaler.inverse_transform(data)

def fill_missing_timesteps(df, time_col, freq, set_type):
    df[time_col] = pd.to_datetime(df[time_col])
    df.sort_values(by=time_col, inplace=True)
    
    first_timestep = df[time_col].iloc[0]
    last_timestep = df[time_col].iloc[-1]
    
    if freq == 'd': # days
        duration =(last_timestep - first_timestep).days
    elif freq == 'h': # hours
        duration =(last_timestep - first_timestep).days * 24
    elif freq == 'a': # years
        duration =np.ceil((last_timestep - first_timestep).days / 365)
    elif freq == 'm': # months
        duration =np.ceil((last_timestep - first_timestep).days / 30)
    elif freq == 'w': # weeks
        duration =np.ceil((last_timestep - first_timestep).days / 7)
    elif freq == 't': # minutes
        duration =np.ceil((last_timestep - first_timestep).seconds / 60)
    else:
        print(f'Error: freq {freq} is not supported.')
        raise NotImplementedError
    
    # return if no missing timesteps
    if duration == df.shape[0]: return
    
    # verbose only for train data 
    if set_type == 0:
        print(f'Data from {first_timestep} to {last_timestep}.')
        
        print(f'{duration} timesteps are present from {df.shape[0]}.')

    # fill missing timesteps
    complete_index = pd.date_range(
        first_timestep, end=last_timestep, 
        # note that the freq is different from self.freq
        # TODO: generalize this
        freq=freq,
        name=time_col
    )
    complete_index = pd.DataFrame(complete_index, columns=[time_col])
    df = df.merge(complete_index, on=time_col, how='right').fillna(method='ffill')
    
    return df

class Dataset_Custom(Dataset):
    def __init__(
        self, args, flag='train', 
        time_col='Date'
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
        self.time_col = time_col
        self.scaler = {}
        self.percent = args.percent
        assert 0<=self.percent<=100, f'Error: percent {self.percent} is not within [0, 100].'
        
        self.__read_data__()

    def __read_data__(self):
        
        df_raw = pd.read_csv(
            os.path.join(self.root_path, self.data_path)
        )
        time_col = self.time_col
        df_raw[time_col] = pd.to_datetime(df_raw[time_col])
        df_raw.sort_values(by=time_col, inplace=True)

        df_raw = fill_missing_timesteps(
            df_raw, time_col, self.freq, self.set_type
        )
        
        # some files have $ in values
        object_columns = df_raw.select_dtypes(include=['object']).columns
        df_raw[object_columns] = df_raw[object_columns].apply(
            lambda x: x.str.replace('$', '')
        )
        df_raw = df_raw.fillna(method='ffill').fillna(method='bfill')

        '''
        df_raw.columns: [time_col, ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(time_col)
        
        if self.features != 'M':
            # only the last input is target
            cols.remove(self.target)
            df_raw = df_raw[[time_col] + cols + [self.target]]
        else:
            # all inputs are targets
            self.target = cols
            df_raw = df_raw[[time_col] + cols]
            
        num_train = int(len(df_raw) * 0.8)
        num_test = int(len(df_raw) * 0.1)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0 and self.args.percent <100:
            percent = self.args.percent
            length = border2 - self.seq_len - self.pred_len - border1
            print(f'Taking {percent}% of {length}.')
            border1 = length * (100-percent)//100 + border1

        # features
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        
        df_stamp = df_raw[[time_col]][border1:border2].reset_index(drop=True)
        df_stamp[time_col] = pd.to_datetime(df_stamp[time_col])
        
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp[time_col].apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp[time_col].apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp[time_col].apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp[time_col].apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop([time_col], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp[time_col].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = self.data_y = df_data[border1:border2].values
        self.data_stamp = data_stamp
        
        # time index of predicted values
        self.index = df_stamp[[time_col]][
            self.seq_len - 1: -self.pred_len
        ].reset_index(drop=True)

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
            self.scaler[index] = scaler
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, scaler, data):
        if not self.scale: return data
        if scaler is None:
            print('Error ! The scaler is None.')
            return data
        
        if len(data.shape)==1:
            data = scaler.inverse_transform(data.reshape(-1, 1))
            return data.flatten()
        elif len(data.shape)>2:
            prev_shape = data.shape
            data = data.reshape(-1, data.shape[-1])
            data = scaler.inverse_transform(data)
            return data.reshape(prev_shape)
        
        return scaler.inverse_transform(data)
    
class MultiTimeSeries(Dataset):
    def __init__(
        self, args, flag='train', time_col='Date'
    ):
        # size [seq_len, label_len, pred_len]
        # info

        self.seq_len = args.seq_len
        self.label_len = args.label_len
        self.pred_len = args.pred_len
        self.flag = flag
        
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        
        self.features = args.features
        self.target = args.target
        self.scale = not args.no_scale
        self.split = (0.8, 0.1, 0.1)
        self.dry_run = args.dry_run
        
        self.args = args
        self.ranges = None
        
        self.group_id = args.group_id
        self.time_col = time_col
        self.time_steps = self.seq_len + self.pred_len
        self.scaler = {}
        self.__read_data__()
        
    def split_data(self, df_raw):
        time_col = self.time_col
        min_year = df_raw[time_col].min()
        
        self.min_year = min_year
        df_raw[time_col] -= min_year
        
        # split the data into train, val, test by time column
        years = sorted(df_raw[time_col].unique())
        train_split, val_split, test_split = self.split

        train_start = 0
        train_end = int(len(years) * train_split)
        val_end = int(len(years) * (train_split + val_split))
        test_end = len(years)

        if self.set_type == 0:
            print('Minimum year ', min_year)
            print(f"Split years:\n \
                Train: {[y+min_year for y in years[train_start: train_end]]}\n \
                Validation: {[y+min_year for y in years[train_end: val_end]]}\n \
                Test: {[y+min_year for y in years[val_end: test_end]]}\n")
        
        # Set borders
        border1s = [0, train_end - self.seq_len, val_end - self.seq_len]
        border2s = [train_end, val_end, test_end]
        
        border1 = years[border1s[self.set_type]]
        border2 = years[border2s[self.set_type]-1]
        
        if self.set_type == 0 and self.args.percent < 100:
            percent = self.args.percent
            length = border2 - self.seq_len - self.pred_len - border1
            if length < 100:
                print(f'Warning: Length {length} is less than 100. Chosing by groups.')
                groups = df_raw[self.group_id].unique()
                choice_size = len(groups) * percent//100
                chosen_groups = np.random.choice(groups, choice_size, replace=False)
                df_raw = df_raw[df_raw[self.group_id].isin(chosen_groups)]
                print(f'Selected {len(chosen_groups)} groups among {len(groups)}.')
            else:
                border1 = length * (100-percent)//100 + border1
                
        self.border1, self.border2 = border1, border2
        
        # filter out data
        df_data = df_raw[
            (df_raw[time_col]>=self.border1) & (df_raw[time_col]<=self.border2)
        ].copy().reset_index(drop=True)
        return df_data
        
    def __read_data__(self):
        # read and clean dataset
        df = pd.read_csv(os.path.join(self.args.root_path,self.args.data_path))
        time_steps = self.time_steps
        
        df.sort_values(by=self.time_col, inplace=True)
        
        df = self.split_data(df)
        df, selected_columns = self.scale_data(df)
        ranges, data, data_stamp = [], [], []
        
        for identifier, df in tqdm(
            df.groupby(self.group_id), desc='Preparing data',
            disable=self.args.disable_progress
        ):
            num_entries = len(df)
            if num_entries < time_steps: continue
            
            for i in range(num_entries - time_steps + 1):
                sliced = df.iloc[i:i + time_steps]
                data.append(sliced[selected_columns])
    
                data_stamp.append(sliced[[self.time_col]].values)
                ranges.append((identifier, i))
            if self.dry_run: break

        self.data = np.array(data)
        self.data_stamp = np.array(data_stamp)      
        self.ranges = ranges
        
    @property    
    def index(self):
        indices = []
        for identifier, start_idx in self.ranges:
            indices.append(
                (identifier, start_idx + self.border1 + self.seq_len + self.min_year)
            )
            
        return pd.DataFrame(
            indices, columns=[self.group_id, self.time_col]
        )
            
    # TODO: add group scaling
    def scale_data(self, df_raw):
        group_id, time_col, target = self.group_id, self.time_col, self.target
        print('Scaling data.')
        # get input features
        input_cols = [
            col for col in df_raw.columns \
                if col not in [group_id, time_col, target]
        ]
        # not all input columns will not be selected for training
        #Important: targets need to be at the end
        if self.features == 'S': selected_columns = [target]
        else: selected_columns = input_cols + [target]
        
        # input feature categories by type
        categorical_features = [
            col for col in selected_columns if df_raw[col].dtype == 'object' or col.endswith('_ID')
        ]
        
        # selected_columns = [col for col in selected_columns if col not in categorical_features]
        self.selected_columns = selected_columns

        numerical_features = [
            col for col in selected_columns if col not in [time_col] + categorical_features + [target]
        ]
        
        # only print one for training data, no need for repetition
        if self.set_type == 0:
            print(f'Categoricals or ID: {categorical_features}.')
            print(f'Numericals: {numerical_features}.')
            print(f'Time column {time_col}, target {target}.')
        
        # not using categorical features for now
        # scaling must be done for categorical features
        if len(categorical_features) > 0:
            # Must scale categorical or object features
            cat_encoder = OrdinalEncoder(dtype=int)
            # fitted on whole dataset to handle unknown labels
            cat_encoder.fit(df_raw[categorical_features])
            df_raw.loc[:, categorical_features] = cat_encoder.transform(
                df_raw[categorical_features]
            )
            
        return df_raw, selected_columns
        
    def __getitem__(self, index):
        s_end = self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data[index][:s_end]
        seq_y = self.data[index][r_begin:r_end]
        
        seq_x_mark = self.data_stamp[index][:s_end]
        seq_y_mark = self.data_stamp[index][r_begin:r_end]
        
        if self.scale:
            if self.features == 'MS':
                scaler = StandardScaler().fit(seq_x[:, :-1])
                seq_x[:, :-1] = scaler.transform(seq_x[:, :-1])
                seq_y[:, :-1] = scaler.transform(seq_y[:, :-1])
                
                scaler = StandardScaler().fit(seq_x[:,-1:])
                seq_x[:, -1:] = scaler.transform(seq_x[:, -1:])
                seq_y[:, -1:] = scaler.transform(seq_y[:, -1:])
            else:
            # assuming inputs are all numerical features
                scaler = StandardScaler().fit(seq_x)

                seq_x = scaler.transform(seq_x) 
                seq_y = scaler.transform(seq_y)
            
            self.scaler[index] = scaler
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data) # - self.seq_len - self.pred_len + 1
    
    def inverse_transform(self, scaler, data):
        if not self.scale: return data
        if scaler is None:
            print('Error ! The scaler is None.')
            return data
        
        if len(data.shape)==1:
            data = scaler.inverse_transform(data.reshape(-1, 1))
            return data.flatten()
        elif len(data.shape)>2:
            prev_shape = data.shape
            data = data.reshape(-1, data.shape[-1])
            data = scaler.inverse_transform(data)
            return data.reshape(prev_shape)
        
        return scaler.inverse_transform(data)