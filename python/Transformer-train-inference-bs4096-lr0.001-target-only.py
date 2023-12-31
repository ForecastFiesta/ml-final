#!/usr/bin/env python
# coding: utf-8

# # LSTM for stock prediction

# ## Import Library

# In[ ]:


#get_ipython().system('sudo pip install torch==2.0.0 pandas numpy tqdm matplotlib')


# In[4]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm

from typing import Any, Dict, List, Tuple

import os
import csv


# In[5]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device


# ## Constants Definition

# In[6]:


EXTENDED_FEATURES = [
]
NORMALIZED_FEATURE = [
]
FILL_ONE_FEATURES = [
]
FILL_MEAN_FEATURES = [
]
MODEL_INPUT_FEATURES = [
    'target'
]


# ## Hyper Parameters

# In[73]:


BATCH_SIZE: int = 4096
EPOCHS: int = 50
LEARNING_RATE: float = 0.001


# ## Import Dataset

# In[39]:


kaggle_dir = './kaggle'
df = pd.read_csv(os.path.join(kaggle_dir, 'train.csv'))
df


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[7]:


def inspect_columns(df):

    result = pd.DataFrame({
        'unique': df.nunique() == len(df),
        'cardinality': df.nunique(),
        'with_null': df.isna().any(),
        'null_pct': round((df.isnull().sum() / len(df)) * 100, 2),
        '1st_row': df.iloc[0],
        'random_row': df.iloc[np.random.randint(low=0, high=len(df))],
        'last_row': df.iloc[-1],
        'dtype': df.dtypes
    })
    return result

inspect_columns(df)

train_df: pd.DataFrame = df.loc[(0 <= df['date_id']) & (df['date_id'] <= 399)]
valid_df: pd.DataFrame = df.loc[(400 <= df['date_id']) & (df['date_id'] <= 477)]
test_df: pd.DataFrame = df.loc[((477 == df['date_id']) & (500 <= df['seconds_in_bucket'])) | ((478 <= df['date_id']) & (df['date_id'] <= 480))]

train_set_x = np.load('./numpy_bin/target_only_train_set_x.npy').astype(np.float32)
train_set_y = np.load('./numpy_bin/target_only_train_set_y.npy').astype(np.float32)
valid_set_x = np.load('./numpy_bin/target_only_valid_set_x.npy').astype(np.float32)
valid_set_y = np.load('./numpy_bin/target_only_valid_set_y.npy').astype(np.float32)

if train_set_y.ndim == 1:
    train_set_y = train_set_y[:, np.newaxis]
if valid_set_y.ndim == 1:
    valid_set_y = valid_set_y[:, np.newaxis]


# In[25]:


print(train_set_x.shape)
print(train_set_y.shape)
print(valid_set_x.shape)
print(valid_set_y.shape)




def PreprocessValidationTestSet(
    raw_df: pd.DataFrame, statistic_result: Dict[int, Dict[str, Dict[str, float]]]
) -> pd.DataFrame:

    data: pd.DataFrame = raw_df.copy()

    # fill out NaN with mean value
    for stock in data['stock_id'].unique().tolist():
        for feat in FILL_MEAN_FEATURES:
            data.loc[
                (data['stock_id'] == stock) & (data[feat].isnull()),
                feat
            ] = statistic_result[stock][feat]['mean']

    # fill out NaN with 1
    for feat in FILL_ONE_FEATURES:
        data[feat] = data[feat].fillna(1.0)

    # normalize features
    for feat in MODEL_INPUT_FEATURES:
        if feat not in NORMALIZED_FEATURE:
            continue
        data['min'] = np.nan
        data['max'] = np.nan

        for stock in data['stock_id'].unique().tolist():
            data.loc[data['stock_id'] == stock, ['min', 'max']] = [
                statistic_result[stock][feat]['min'],
                statistic_result[stock][feat]['max']
            ]

        data[feat] = (data[feat] - data['min']) / (data['max'] - data['min'])
        data = data.drop('min', axis=1)
        data = data.drop('max', axis=1)

    data.dropna(inplace=True)

    print(f'Data has nan: \n{data.isnull().any()}')

    return data

# In[12]:

def GenPreprocessInfo(raw_df: pd.DataFrame) -> Dict[int, Dict[str, Dict[str, float]]]:
    '''
    Return value example:
    {
        'STOCK_ID': {
            'COLUMN': {
                'min': 0.0,
                'max': 1.0,
                'mean': 0.5
            }
        }
    }
    '''
    data: pd.DataFrame = raw_df.copy()
    statistic_result: Dict[int, Dict[str, Dict[str, float]]] = {}

    for stock in data['stock_id'].unique().tolist():
        stock_df: pd.DataFrame = data.loc[data['stock_id'] == stock]
        stock_info: Dict[str, Dict[str, float]] = {}

        for feat in stock_df.columns:
            if pd.api.types.is_string_dtype(stock_df[feat]):
                continue
            stock_feat_info: Dict[str, float] = {}
            stock_feat_info['min'] = stock_df[feat].min(skipna=True)
            stock_feat_info['max'] = stock_df[feat].max(skipna=True)
            stock_feat_info['mean'] = stock_df[feat].mean(skipna=True)
            stock_info[feat] = stock_feat_info

        statistic_result[stock] = stock_info

    return statistic_result

preprocess_info = GenPreprocessInfo(raw_df=train_df)

valid_pp_df = PreprocessValidationTestSet(raw_df=valid_df, statistic_result=preprocess_info)
valid_pp_df


# In[44]:



test_pp_df = PreprocessValidationTestSet(raw_df=test_df, statistic_result=preprocess_info)
test_pp_df


# In[68]:


train_set_x_torch, train_set_y_torch, valid_set_x_torch, valid_set_y_torch = (
    torch.tensor(train_set_x).clone().detach(),
    torch.tensor(train_set_y).clone().detach(),
    torch.tensor(valid_set_x).clone().detach(),
    torch.tensor(valid_set_y).clone().detach()
)

train_loader = DataLoader(TensorDataset(train_set_x_torch, train_set_y_torch), shuffle=True, batch_size=BATCH_SIZE)
valid_loader = DataLoader(TensorDataset(valid_set_x_torch, valid_set_y_torch), shuffle=True, batch_size=BATCH_SIZE)


# ## Model Definition

# In[69]:


class Transformer_Model(nn.Module):
    def __init__(self, feature_num, d_model, nhead, num_layers):
        super(Transformer_Model, self).__init__()
        self.embedding = nn.Linear(feature_num, d_model)
        self.tf1 = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.5)
        self.tf2 = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers, batch_first=True)
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.tf1.encoder(x)
        x = x[:, -1, :]
        x = self.fc(x)
        x = self.dropout(x)
        x = self.tf2.encoder(x)
        x = self.decoder(x)

        return x



class LSTM_Model(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1
        )
        self.linear = nn.Linear(64 * 2, 1)
        
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x[:, -1, :])
        return x


# ## Training

# In[70]:


def train(
    model, train_loader: DataLoader, valid_loader: DataLoader, optimizer, loss_fn, epoch_cnt: int
) -> Tuple[List[float], List[float]]:
    train_loss_list: List[float] = []
    valid_loss_list: List[float] = []
    
    print(f'The batch size of train_loader = {train_loader.batch_size}')
    print(f'The batch size of valid_loader = {valid_loader.batch_size}')

    min_loss: float = 10.0
    
    for epoch in range(epoch_cnt):
        train_batch_loss_list: List[float] = []
        valid_batch_loss_list: List[float] = []
        
        model.train()
        pbar = tqdm(train_loader)
        pbar.set_description(f"Training Epoch {epoch}")
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            pred_y = model(batch_x)
            loss = loss_fn(pred_y, batch_y)
            train_batch_loss_list.append(loss.cpu().detach().numpy())
            
            loss.backward()
            optimizer.step()

            if len(train_batch_loss_list) % 1000 == 999:
                train_loss_list.append(float(np.mean(train_batch_loss_list[-100:])))
        
        model.eval()
        better_model: bool = False
        with torch.no_grad():
            pbar = tqdm(valid_loader)
            pbar.set_description(f"Validating Epoch {epoch}")
            for batch_x, batch_y in pbar:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                pred_y = model(batch_x)
                loss = loss_fn(pred_y, batch_y)
                valid_batch_loss_list.append(loss.cpu().detach().numpy())

                if len(valid_loss_list) == 0 or float(np.mean(valid_batch_loss_list)) < min_loss:
                    better_model = True
                    min_loss = float(np.mean(valid_batch_loss_list))
                valid_loss_list.append(float(np.mean(valid_batch_loss_list)))
        print (better_model)
        print (np.mean(valid_batch_loss_list))
        print (min_loss)
        if better_model:
            torch.save(model, f"./models/transformer-{BATCH_SIZE}-lr{LEARNING_RATE}-targetonly.pt")
        
        print("Epoch %d: train_loss %.4f, val_loss %.4f" % (epoch, np.mean(train_batch_loss_list), np.mean(valid_batch_loss_list)))
        
    return train_loss_list, valid_loss_list


# In[71]:


model = Transformer_Model(feature_num=len(MODEL_INPUT_FEATURES), d_model=64, nhead=8, num_layers=1).to(device)
model.to(device)


# In[72]:


optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.L1Loss()


# In[ ]:


train_loss_list: List[float] = []
valid_loss_list: List[float] = []
train_loss_list, valid_loss_list = train(
    model=model, 
    train_loader=train_loader, 
    valid_loader=valid_loader, 
    optimizer=optimizer, 
    loss_fn=loss_fn, 
    epoch_cnt=EPOCHS
)


# In[59]:


# loss graph
fig = plt.figure(figsize=(45, 5))
ax1 = fig.add_subplot(1, 2, 1)

train_loss_x_axis: List[int] = np.arange(len(train_loss_list)).tolist()
valid_loss_x_axis: List[int] = np.arange(start=len(train_loss_list)-1, stop=-1, step=(len(train_loss_list)/len(valid_loss_list))*-1, dtype=np.float32).astype(np.int32).tolist()[::-1]

ax1.set_title('Loss')
ax1.plot(train_loss_x_axis, train_loss_list, marker='.')
ax1.plot(valid_loss_x_axis, valid_loss_list, marker='.')
ax1.legend(['train_loss', 'valid_loss'], loc='upper left')
ax1.set_xlabel('1000 Batch')
plt.show()


# In[52]:


torch.save(model, f"./models/lstm-bs{BATCH_SIZE}-lr{LEARNING_RATE}.pt")


# ## Testing

# In[60]:


class TimeSeriesDataset(Dataset):
    def __init__(self, raw_df: pd.DataFrame, window_size: int) -> None:
        super().__init__()
        self.data: pd.DataFrame = raw_df.copy()
        self.window_size: int = window_size
        self.stock_dfs: Dict[int, pd.DataFrame]
        self.idx_map: Dict[int, Dict[str, int]]
        self.stock_dfs, self.idx_map = TimeSeriesDataset._SplitStockDataFrame(raw_df=self.data, window_size=self.window_size)
    

    def __getitem__(self, _index: int) -> Tuple[np.ndarray, np.ndarray, int, int, int]:
        '''
        return x, y, date_id, seconds_in_buckets, stock_id
        '''
        index: int = _index
        stock: int = self.idx_map[index]['stock']
        stock_idx: int = self.idx_map[index]['idx']

        start_idx, end_idx = stock_idx - self.window_size, stock_idx + 1
        return TimeSeriesDataset._GenTimeSeriesData(raw_df=self.stock_dfs[stock].iloc[start_idx:end_idx])


    def __len__(self) -> int:
        return len(self.idx_map)
    
    def FillTarget(self, stock: int, date: int, second: int, value: np.float32) -> None:
        if stock not in self.stock_dfs:
            return
        self.stock_dfs[stock].loc[(self.stock_dfs[stock]['date_id'] == date) & (self.stock_dfs[stock]['seconds_in_bucket'] == second), 'pred_target'] = value
    
    @staticmethod
    def _GenTimeSeriesData(raw_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, int, int, int]:
        '''
        raw_df: The rows of time t-n ~ t for predicting the target of t.
        The generated features contains the data of time t-n ~ t-1.
        return x, y, date_id, seconds_in_buckets, stock_id
        '''
        df = raw_df.copy()
        df = df.sort_values(by=['date_id', 'seconds_in_bucket'], ascending=True)

        date: int = df['date_id'].tolist()[-1]
        second: int = df['seconds_in_bucket'].tolist()[-1]
        stock: int = df['stock_id'].tolist()[-1]
        y: np.float32 = df['target'].tolist()[-1]

        pred_target_not_null_indices = df['pred_target'].notnull()
        df.loc[pred_target_not_null_indices, 'target'] = df.loc[pred_target_not_null_indices, 'pred_target']
        
        df = df[MODEL_INPUT_FEATURES]
        df_numpy: np.ndarray = df.to_numpy()
        
        x: np.ndarray = df_numpy[:-1, :]
        
        return x.astype(np.float32), np.array(y).astype(np.float32), date, second, stock
    
    @staticmethod
    def _SplitStockDataFrame(raw_df: pd.DataFrame, window_size: int) -> Tuple[Dict[int, pd.DataFrame], Dict[int, Dict[str, int]]]:
        stock_dfs: Dict[int, pd.DataFrame] = {}
        idx_map: Dict[int, Dict[str, int]] = {}

        data: pd.DataFrame = raw_df.copy()
        data = data.sort_values(by=['date_id', 'seconds_in_bucket', 'stock_id'], ascending=True)
        data.reset_index(drop=True, inplace=True)
        data_idx_offset: int = 5 * 200

        for stock in data['stock_id'].unique().tolist():
            stock_df: pd.DataFrame = data.copy().loc[data['stock_id'] == stock]
            stock_df.insert(stock_df.shape[1], 'new_idx', range(stock_df.shape[0]))
            stock_df.insert(stock_df.shape[1], 'pred_target', np.nan)
            for idx, row in stock_df.iterrows():
                if row['new_idx'] < window_size:
                    continue
                stock_idx_map: Dict[str, int] = {'stock': stock, 'idx': row['new_idx']}
                idx_map[int(idx) - data_idx_offset] = stock_idx_map
            stock_df.reset_index(drop=True, inplace=True)
            stock_df.drop('new_idx', axis=1, inplace=True)
            stock_dfs[stock] = stock_df

        return stock_dfs, idx_map


# In[61]:


test_dataset: TimeSeriesDataset = TimeSeriesDataset(raw_df=test_pp_df, window_size=5)


# In[62]:


print(test_dataset[0])


# In[97]:


# test_dataset.FillTarget(stock=0, date=477, second=540, value=np.float32(12))
# print(test_dataset[0])


# In[63]:


def test(
    model, test_set: TimeSeriesDataset
) -> pd.DataFrame:
    
    result: pd.DataFrame = pd.DataFrame(columns=['date_id', 'seconds_in_bucket', 'stock_id', 'time_id', 'row_id', 'target'])

    model.eval()
    with torch.no_grad():
        with tqdm(total=len(test_set)) as pbar:
            pbar.set_description(f"Testing")
            for iter in range(len(test_set)):
                batch_x, batch_y, date, second, stock = test_set[iter]

                batch_x = torch.from_numpy(batch_x[np.newaxis, :]).to(device)
                batch_y = torch.from_numpy(batch_y).to(device)
                
                pred_y = model(batch_x).cpu().detach().numpy()

                for idx in range(len(pred_y)):
                    test_set.FillTarget(stock=stock, date=date, second=second, value=pred_y[idx])
                    curr_result: pd.DataFrame = pd.DataFrame.from_dict(data={
                        'date_id': [date], 
                        'seconds_in_bucket': [second], 
                        'stock_id': [stock], 
                        'time_id': [int(26290 + (date - 478) * 55 + second)], 
                        'row_id': [f'{date}_{second}_{stock}'], 
                        'target': [pred_y[idx][0]]
                    })
                    if result.shape[0] > 0:
                        result = pd.concat([result, curr_result], ignore_index=True)
                    else:
                        result = curr_result
                pbar.update(1)
        
    return result


# In[64]:


test_result = test(model=model, test_set=test_dataset)


# In[65]:


test_result


# In[66]:


test_out_result = test_result.sort_values(by=['date_id', 'seconds_in_bucket', 'stock_id'], ascending=True)
test_out_result.drop(['date_id', 'seconds_in_bucket', 'stock_id'], axis=1, inplace=True)
test_out_result.to_csv('./submission.csv', index=False)


# In[ ]:




