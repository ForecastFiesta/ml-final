
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




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device


# ## Constants Definition

# In[6]:

MODEL_PATH = '/home/computing/pine0113/optiver-2023-pine0113/models/transformer-16-lr2.5e-05.pt'


kaggle_dir = './kaggle'
df = pd.read_csv(os.path.join(kaggle_dir, 'train.csv'))
df



EXTENDED_FEATURES = [
]
NORMALIZED_FEATURE = [
    'imbalance_size',
    'matched_size',
    'bid_size',
    'ask_size'
]
FILL_ONE_FEATURES = [
    "reference_price",
    "far_price",
    "near_price",
    "bid_price",
    "ask_price",
    "wap"
]
FILL_MEAN_FEATURES = [
    "imbalance_size",
    "matched_size"
]
MODEL_INPUT_FEATURES = [
    'imbalance_size',
    'imbalance_buy_sell_flag',
    'reference_price',
    'matched_size',
    'far_price',
    'near_price',
    'bid_price',
    'bid_size',
    'ask_price',
    'ask_size',
    'wap',
    'target'
]


# In[5]:


df.info()


# In[6]:


df.isnull().sum()

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


test_pp_df = PreprocessValidationTestSet(raw_df=test_df, statistic_result=preprocess_info)
test_pp_df


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

print(test_dataset[0])
print(test_dataset[1])


def test(
    model, test_set: TimeSeriesDataset
) -> pd.DataFrame:

    result: pd.DataFrame = pd.DataFrame(columns=['date_id', 'seconds_in_bucket', 'stock_id', 'time_id', 'row_id', 'target'])
    
    model.eval()
    with torch.no_grad():
        with tqdm(total=len(test_set)) as pbar:
            pbar.set_description(f"Testing")
                
            for iter in range(len(test_set)):
#                print("iter:")
#                print(len(test_set))
#                print(test_set)

                batch_x, batch_y, date, second, stock = test_set[iter]

                batch_x = torch.from_numpy(batch_x[np.newaxis, :]).to(device)
                batch_y = torch.from_numpy(batch_y).to(device)

                pred_y = model(batch_x).cpu().detach().numpy()

                for idx in range(len(pred_y)):
#                    print(pred_y,idx)
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



#model = Transformer_Model(feature_num=len(MODEL_INPUT_FEATURES), d_model=64, nhead=8, num_layers=1).to(device)
model = torch.load(MODEL_PATH)
model.to(device)


test_result = test(model=model, test_set=test_dataset)


test_out_result = test_result.sort_values(by=['date_id', 'seconds_in_bucket', 'stock_id'], ascending=True)
test_out_result.drop(['date_id', 'seconds_in_bucket', 'stock_id','time_id','row_id'], axis=1, inplace=True)
print(test_out_result)

df = pd.read_csv("./kaggle/train.csv")
ans_df = df.loc[df['date_id'] >= 478]
ans = ans_df['target'].to_numpy()

print(ans)



# In[19]:


our_result = test_out_result['target'].to_numpy()


# In[6]:


def mae(x: np.ndarray, y: np.ndarray):
    if np.shape(x) != np.shape(y):
        print(np.shape(x),np.shape(y))
        raise RuntimeError

    return np.sum(np.absolute(y - x)) / np.shape(x)



print(mae(our_result, ans))


