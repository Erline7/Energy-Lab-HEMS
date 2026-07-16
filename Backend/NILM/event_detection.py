import os
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# cusum事件检测
appliances = {
    'CDE': {'k': 1000, 'h': 4000, 'true_off': 20},   # k为底噪波动，h用于确认开关
    'WOE': {'k': 50, 'h': 1500, 'true_off': 10},
    'HPE': {'k': 25, 'h': 60, 'true_off': 30},
    'CWE': {'k': 30, 'h': 200, 'true_off': 20},
    'FGE': {'k': 50, 'h': 100, 'true_off': 40},
    'BME': {'k': 20, 'h': 150, 'true_off': 10},
    'TVE': {'k': 30, 'h': 150, 'true_off': 30}
}

def read_file(filepath: str, appliance_name: str) -> pd.Series:
    df = pd.read_csv(filepath, sep=',', usecols=['unix_ts', 'P'])
    df['time'] = pd.to_datetime(df['unix_ts'], unit='s')
    column = df.set_index('time')['P']
    column.name = appliance_name
    return column

def clean(df: pd.DataFrame) -> pd.DataFrame:
    if df.isnull().sum().sum() > 0:   # isnull检测缺失值，统计每列，以及总的缺失值
        df =df.ffill().bfill()
    if (df < 0).sum().sum() > 0:
        df = df.clip(lower=0)   # 裁剪负值
    return df

def load(data_dir: str, appliances: list = None):
    all_series = []
    for name in appliances:
        fpath = os.path.join(data_dir, f'Electricity_{name}.csv')
        if not os.path.isfile(fpath):
            print(f'File {fpath} does not exist')
            continue
        s = read_file(fpath, name)
        all_series.append(s)
        print(f'Loaded {name}, 记录数={len(s)}')
    df = pd.concat(all_series, axis = 1, join = 'inner')
    return df

def split_data(raw, train=0.7, val=0.1):
    T = len(raw)
    i1 = int(T * train)
    i2 = int(T * (train + val))
    train_raw, val_raw, test_raw = raw[:i1], raw[i1:i2], raw[i2:]
    return train_raw, val_raw, test_raw, i1, i2

def features(df: pd.DataFrame, appliance_name: str = 'DEFAULT') -> pd.DataFrame:
    idx = df.index
    hour = idx.hour + idx.minute / 60.0


    df = df.copy()
    df['sin_hour'] = np.sin(2 * np.pi * hour / 24).astype(np.float32)
    df['cos_hour'] = np.cos(2 * np.pi * hour / 24).astype(np.float32)
    df['is_weekend'] = (idx.dayofweek >= 5).astype(np.float32)
    if appliance_name == 'HPE':
        month = idx.month
        df['cos_month'] = np.cos(2 * np.pi * (month - 1) / 12).astype(np.float32)

    return df


def normalize(train_raw, val_raw, test_raw, n_power_cols: int = None):
    scaler = StandardScaler()
    if n_power_cols is None or n_power_cols == train_raw.shape[1]:
        scaler.fit(train_raw)
        return scaler, scaler.transform(train_raw), scaler.transform(val_raw), scaler.transform(test_raw)
    scaler.fit(train_raw[:, :n_power_cols])     # 只对功率列拟合

    def _trans(x):
        out = x.copy()
        out[:, :n_power_cols] = scaler.transform(x[:, :n_power_cols])
        return out
    return scaler, _trans(train_raw), _trans(val_raw), _trans(test_raw)

def cusum(power, appliance_name):
    config = appliances.get(appliance_name, {'k': 20, 'h': 50})
    k, h = config['k'], config['h']
    T = len(power)
    sl = [0] * T
    sr = [0] * T
    events = []
    window_size = 4
    cold_win = 0
    for i in range(window_size, T):
        if cold_win > 0:
            cold_win -= 1
            continue
        pre_window = power[i - window_size: i]
        pre_mean = np.mean(pre_window)
        diff = power[i] - pre_mean
        sl[i] = max(0, sl[i - 1] + diff - k)    # for h, 每次的值相加但去掉k
        sr[i] = max(0, sr[i - 1] - diff - k)
        if sl[i] > h:
            if diff > k:
                events.append({'index': i-1, 'delta_p': diff, 'type': 'START'})
                sl[i], sr[i] = 0, 0
                cold_win = window_size//2
        elif sr[i] > h:
            if -diff > k:
                events.append({'index': i-1, 'delta_p': diff, 'type': 'END'})
                sl[i],sr[i] = 0, 0;
                cold_win = window_size//2
    return events

def extract_cusum(power, appliance_name):
    events = cusum(power, appliance_name)
    is_active = np.zeros(len(power), dtype=bool)
    current_state = False # 开关
    last_idx = 0
    for a in events:
        idx = a['index']
        if a['type'] == 'START' and not current_state:
            current_state = True
            last_idx = idx
        elif a['type'] == 'END' and current_state:
            is_active[last_idx:idx] = True
            current_state = False
    if current_state: # 如果最终状态是开就把他到最后时间都记为开
        is_active[last_idx:] = True
    return power[is_active].reshape(-1, 1)

def labels(power, gmm, remap, true_off):
    label_list = [0] * len(power)
    active = []
    for i in range(len(power)):
        if power[i] >= true_off:
            active.append(i)
    if not active:
        return np.array(label_list)
    if active:
        power_active = []
        for i in active:
            power_active.append([power[i]])  # 变成多维数组，是gmm需要的格式
        pred = gmm.predict(power_active)

        # 转换成最终标签
        for j in range(len(active)):
            current_position = active[j]
            current_prediction = pred[j]
            if current_prediction in remap:
                mapped_value = remap[current_prediction]
                if mapped_value == -1:
                    label_list[current_position] = 0
                else:
                    label_list[current_position] = mapped_value + 1
            else:
                label_list[current_position] = 0
    return np.array(label_list)


def fit_gmm(train: np.ndarray, all_data: np.ndarray, appliance_name: list = None, n_range: tuple = (2,6), max_sample: int = 10000):
    Time, N = all_data.shape    # all_raw是二维矩阵
    label_array = np.empty((Time, N), dtype=np.int64) # 创建64位空数组
    label_array[:] = 0
    n_states = []
    if appliance_name is None:
        appliance_name = ['DEFAULT'] * N
    for i in range(N):
        col_train = train[:, i]
        col_all = all_data[:, i]
        if i < len(appliance_name):
            name = appliance_name[i]
        else:
            name = 'DEFAULT'
        print(f'**************电器{name}**************')
        if col_train.std() < 1e-3:
            n_states.append(1)
            continue
        active_train = extract_cusum(col_train, name)   # 创建开机的多维数组
        config = appliances.get(name)
        if config is None:
            config = {'k': 20, 'true_off': 10}
        off_thresh = config['k']
        true_off = config.get('true_off', off_thresh)
        if len(active_train) < 50:
            print(f'WARNING: 数据不足！')
            active_train = col_train
        if len(active_train) > max_sample:
            np.random.seed(42)
            idx = np.random.choice(len(active_train),max_sample, replace=False)
            col_sample = active_train[idx]
        else:
            col_sample = active_train
        best_n, best_score = n_range[0], np.inf
        #   找最好的n
        for n in range(n_range[0], n_range[1] + 1):
            gmm = GaussianMixture(
                n_components=n,
                covariance_type='full',
                random_state=42)
            gmm.fit(col_sample)   # MM学习数据
            score = gmm.bic(col_sample)   #计算BIC分数
            if score < best_score:
                best_score, best_n = score, n
        gmm_final = GaussianMixture(
            n_components=best_n,
            covariance_type='full',
            random_state=42
        )
        gmm_final.fit(active_train)
        power = gmm_final.means_.flatten() # 找到每个状态的平均功率值
        power_idx = np.argsort(power) # 索引从小到大排序
        power_sorted = np.sort(power)
        max_power = float(power_sorted.max())

        merge = max(max_power * 0.07, 20.0) # 合并相近的状态
        current_group = [0] # 从状态一开始
        merge_group = [] # 放合并后的组
        for n in range(1, best_n):
            gap = power_sorted[n] - power_sorted[n - 1]
            if gap < merge:
                current_group.append(n) #
            else:
                merge_group.append(current_group)
                current_group = [n]
        merge_group.append(current_group)
        final_group = []
        off = []
        for k in merge_group:
            group_mean = power_sorted[k].mean() # 计算索引列表均值
            if group_mean < true_off:
                off.extend(k)
                print(f"标记为关,{group_mean:.1f}W")
            else:
                final_group.append(k)
                if group_mean < off_thresh:
                    print(f"标记为待机,{group_mean:.1f}W")
                else:
                    print(f"标记为运行,{group_mean:.1f}W")
        remap = {} # 创建映射表
        for m, n in enumerate(final_group):
            for a in n:
                remap[int(power_idx[a])] = m
        for m in off:
            remap[int(power_idx[m])] = -1
        col_labels = labels(col_all, gmm_final, remap, true_off)
        label_array[:, i] = col_labels
        total = len(final_group) + 1
        n_states.append(total)
        print(f'    GMM最优k={best_n}  总状态数={total}（含OFF）  off_thresh={off_thresh}W')
    return label_array, n_states
if __name__ == '__main__':
    pass