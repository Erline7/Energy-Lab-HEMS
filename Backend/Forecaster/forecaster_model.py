import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

APPLIANCES = ['HPE', 'BME', 'CDE', 'WOE', 'TVE']

def count_duration(state_arr: np.ndarray):
    duration  = np.zeros(len(state_arr), dtype=np.float32)
    cnt  = 0
    prev = -1
    for i, j in enumerate(state_arr):
        if j > 0 and j == prev:
            cnt += 1
        elif j > 0:
            cnt = 1
        else:
            cnt = 0
        duration[i] = cnt
        prev   = int(j)
    return duration

class Energy_dataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, weather_csv_path=None,
                 history_len=96, future_len=96,
                 max_power=None, mode='refined'):
        self.history_len = history_len
        self.future_len  = future_len
        self.mode        = mode

        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        df.index.name = 'Time'
        df.index = pd.to_datetime(df.index)

        state_cols = [f'{a}_State' for a in APPLIANCES if f'{a}_State' in df.columns]
        power_cols = ['WHE_True']
        keep_cols  = [c for c in power_cols + state_cols if c in df.columns]

        def mode_or_zero(series):
            return int(series.mode().iloc[0]) if len(series) > 0 else 0

        agg_dict = {c: 'mean' for c in power_cols if c in df.columns}
        for c in state_cols:
            agg_dict[c] = mode_or_zero

        df = df[keep_cols].resample('15min').agg(agg_dict).dropna(subset=['WHE_True'])
        df = df.bfill().ffill()

        if max_power is None:
            self.max_power = float(df['WHE_True'].max())
        else:
            self.max_power = max_power
            
        if 'HPE_State' in df.columns:
            df['HPE_duration'] = count_duration(df['HPE_State'].values)
            df['HPE_duration'] = (df['HPE_duration'] / 96.0).clip(upper=1.0)
        if 'CDE_State' in df.columns:
            df['CDE_duration'] = count_duration(df['CDE_State'].values)
            df['CDE_duration'] = (df['CDE_duration'] / 96.0).clip(upper=1.0)

        for w in [2, 4, 12]:
            df[f'roll_max_{w}'] = df['WHE_True'].rolling(window=w, min_periods=1).max()

        if weather_csv_path and os.path.exists(weather_csv_path):
            w_df = pd.read_csv(weather_csv_path)
            w_df['Time'] = pd.to_datetime(w_df['Date/Time'])
            w_df = w_df[['Time', 'Temp (C)', 'Rel Hum (%)']].set_index('Time')
            w_df = w_df.resample('15min').ffill()
            w_df.columns = ['Temperature', 'Humidity']
            df = df.join(w_df, how='left').ffill().fillna(0)
        else:
            df['Temperature'] = 0.0
            df['Humidity']    = 0.0

        # 計算所有時間特徵
        df['sin_hour']    = np.sin(2 * np.pi * df.index.hour / 24.0)
        df['cos_hour']    = np.cos(2 * np.pi * df.index.hour / 24.0)
        df['is_weekend']  = (df.index.dayofweek >= 5).astype(int)
        df['sin_month']   = np.sin(2 * np.pi * (df.index.month - 1) / 12.0)
        df['cos_month']   = np.cos(2 * np.pi * (df.index.month - 1) / 12.0)
        df['hour_of_day'] = df.index.hour / 24.0
        df['day_of_week'] = df.index.dayofweek / 7.0

        for lag in [1, 2, 3, 4, 12]:
            df[f'lag_{lag}'] = df['WHE_True'].shift(lag)
        df['diff_1'] = df['WHE_True'].diff().fillna(0)
        df.dropna(inplace=True)

        self.whe_power = (df[['WHE_True']].values / self.max_power).astype(np.float32)

        if self.mode == 'full':
            base_cols = [
                'sin_hour', 'cos_hour', 'is_weekend',
                'sin_month', 'cos_month', 'hour_of_day', 'day_of_week',
                'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_12',
                'diff_1',
                'roll_max_2', 'roll_max_4', 'roll_max_12',
                'Temperature', 'Humidity'
            ]
        else: 
            base_cols = [
                'sin_hour', 'cos_hour', 'sin_month', 'cos_month', 'is_weekend',
                'lag_1', 'lag_2', 'lag_3', 'lag_4',
                'diff_1',
                'roll_max_2', 'roll_max_4', 'roll_max_12',
                'Temperature', 'Humidity'
            ]

        base_cols  = [c for c in base_cols if c in df.columns]
        extra_data = df[base_cols].values.astype(np.float32)

        for i, col in enumerate(base_cols):
            if 'lag' in col or 'diff' in col or 'roll' in col:
                extra_data[:, i] /= self.max_power
            elif col == 'Temperature':
                extra_data[:, i] = (extra_data[:, i] + 20.0) / 60.0
            elif col == 'Humidity':
                extra_data[:, i] /= 100.0

        nilm_data_list      = []
        self.nilm_cols_used = []

        for col in state_cols:
            if col in df.columns:
                raw       = df[col].values.astype(np.float32)
                max_state = max(float(raw.max()), 1.0)
                nilm_data_list.append((raw / max_state).reshape(-1, 1))
                self.nilm_cols_used.append(col)

        if 'HPE_duration' in df.columns:
            nilm_data_list.append(df['HPE_duration'].values.reshape(-1, 1))
            self.nilm_cols_used.append('HPE_duration')
        if 'CDE_duration' in df.columns:
            nilm_data_list.append(df['CDE_duration'].values.reshape(-1, 1))
            self.nilm_cols_used.append('CDE_duration')
            
        if nilm_data_list:
            nilm_data = np.concatenate(nilm_data_list, axis=1)
            self.all_features = np.concatenate(
                [self.whe_power, extra_data, nilm_data], axis=1
            )
        else:
            self.all_features = np.concatenate(
                [self.whe_power, extra_data], axis=1
            )

        self.input_dim = self.all_features.shape[1]
        self.n_samples = len(df) - history_len - future_len + 1

        print(f"mode='{mode}' | input_dim={self.input_dim} | n_samples={self.n_samples:,}")
        print(f"基础特征 ({len(base_cols)}列): {base_cols}")
        print(f"NILM特征 ({len(self.nilm_cols_used)}列): {self.nilm_cols_used}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return (
            torch.FloatTensor(
                self.all_features[index: index + self.history_len]
            ),
            torch.FloatTensor(
                self.whe_power[index + self.history_len:
                               index + self.history_len + self.future_len]
            )
        )

class ResidualTCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.match_dim = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        return self.act(self.match_dim(x) + self.bn(self.conv(x)))

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out):
        weights = torch.softmax(self.attn(lstm_out), dim=1)
        context = (weights * lstm_out).sum(dim=1, keepdim=True)
        context = context.expand_as(lstm_out)
        return torch.cat([lstm_out, context], dim=-1)

class WHEModel(nn.Module):
    def __init__(self, input_dim, n_nilm_features, cnn_filters=16, hidden_dim=256, future_len=96):
        super().__init__()
        self.future_len = future_len
        self.n_nilm_features = n_nilm_features
        self.tcn = nn.Sequential(
            ResidualTCNBlock(input_dim, cnn_filters, 1),
            ResidualTCNBlock(cnn_filters, cnn_filters, 2),
            ResidualTCNBlock(cnn_filters, cnn_filters, 4),
        )
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True, dropout=0.3
        )
        self.temporal_attn = TemporalAttention(hidden_dim)
        combined_dim = (hidden_dim * 2) + cnn_filters + self.n_nilm_features
        self.fc = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, future_len * 3)
        )

    def forward(self, x):
        B = x.size(0)
        tcn_in = x.permute(0, 2, 1)
        tcn_out = self.tcn(tcn_in).permute(0, 2, 1)
        tcn_pool = tcn_out.mean(dim=1)
        lstm_out, _ = self.lstm(x)
        attn_out = self.temporal_attn(lstm_out)
        attn_last = attn_out[:, -1, :]
        current_nilm_states = x[:, -1, -self.n_nilm_features:]
        combined = torch.cat([attn_last, tcn_pool, current_nilm_states], dim=-1)
        out = self.fc(combined)
        out = out.view(B, self.future_len, 3)
        p10 = out[..., 0:1]
        p50 = p10 + F.softplus(out[..., 1:2])
        p90 = p50 + F.softplus(out[..., 2:3])
        return F.softplus(torch.cat([p10, p50, p90], dim=-1))