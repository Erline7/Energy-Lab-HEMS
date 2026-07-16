import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.compose import TransformedTargetRegressor
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
import joblib
import warnings
warnings.filterwarnings('ignore')

def train_gas_model():

    df = pd.read_csv('../datasets/NaturalGas_Monthly.csv')

    X = df[['Avg Outside Temp (℃)']].values
    y = df['Net Consumption (GJ)'].values

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)
    print(f"燃气模型 R2 得分: {r2_score(y, y_pred):.2f}")
    print(f"燃气模型 MAE 误差: {mean_absolute_error(y, y_pred):.2f} GJ")

    joblib.dump(model, '../checkpoints/gas_model.pkl')
    print("燃气模型已保存为 gas_model.pkl\n")

def load_and_resample_water(filepath, col_name):
    df = pd.read_csv(filepath)
    df['time'] = pd.to_datetime(df['unix_ts'], unit='s')
    df.set_index('time', inplace=True)
    hourly = df.resample('H')['avg_rate'].sum().reset_index()
    hourly.rename(columns={'avg_rate': col_name}, inplace=True)
    return hourly

def train_water_model_expert():
    print("训练多维度水务预测")
    print("合并子水表数据")
    df_whw = load_and_resample_water('../datasets/Water_WHW.csv', 'WHW_rate')
    df_htw = load_and_resample_water('../datasets/Water_HTW.csv', 'HTW_rate')
    df_dww = load_and_resample_water('../datasets/Water_DWW.csv', 'DWW_rate')

    df_water = df_whw.merge(df_htw, on='time', how='inner').merge(df_dww, on='time', how='inner')
    df_water.fillna(0, inplace=True)

    df_water['hour'] = df_water['time'].dt.hour
    df_water['hour_sin'] = np.sin(2 * np.pi * df_water['hour'] / 24)
    df_water['hour_cos'] = np.cos(2 * np.pi * df_water['hour'] / 24)
    df_water['dayofweek'] = df_water['time'].dt.dayofweek
    df_water['is_weekend'] = (df_water['dayofweek'] >= 5).astype(int)

    df_water['is_morning_peak'] = df_water['hour'].apply(lambda x: 1 if 7 <= x <= 9 else 0)
    df_water['is_evening_peak'] = df_water['hour'].apply(lambda x: 1 if 19 <= x <= 22 else 0)
    df_water['is_sleep_time'] = df_water['hour'].apply(lambda x: 1 if 1 <= x <= 5 else 0)

    for col in ['WHW_rate', 'HTW_rate', 'DWW_rate']:
        df_water[f'{col}_lag1'] = df_water[col].shift(1)    # 上一小时的用水量
        df_water[f'{col}_lag24'] = df_water[col].shift(24)  # 昨天同一小时的用水量
        df_water[f'{col}_roll3'] = df_water[col].shift(1).rolling(window=3).mean() # 过去3小时滑动平均

    df_water.dropna(inplace=True)

    feature_cols = [
        'hour_sin', 'hour_cos', 'dayofweek', 'is_weekend',
        'is_morning_peak', 'is_evening_peak', 'is_sleep_time',
        'WHW_rate_lag1', 'HTW_rate_lag1', 'DWW_rate_lag1',
        'WHW_rate_lag24', 'HTW_rate_lag24', 'DWW_rate_lag24',
        'WHW_rate_roll3', 'HTW_rate_roll3', 'DWW_rate_roll3'
    ]
    target_cols = ['WHW_rate', 'HTW_rate', 'DWW_rate']

    X = df_water[feature_cols].values
    y = df_water[target_cols].values

    split_idx = int(len(df_water) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    base_xgb = XGBRegressor(
        n_estimators=800,        # 树的数量
        learning_rate=0.03,      # 学习率
        max_depth=6,             # 树的深度
        subsample=0.8,           # 样本随机采样率
        colsample_bytree=0.8,    # 特征随机采样率
        random_state=42,
        n_jobs=-1
    )

    log_transformed_model = TransformedTargetRegressor(
        regressor=base_xgb,
        func=np.log1p,
        inverse_func=np.expm1
    )
    multi_xgb = MultiOutputRegressor(log_transformed_model)

    multi_xgb.fit(X_train, y_train)

    y_pred = multi_xgb.predict(X_test)
    print(f"评估完成！各项真实误差 (MAE):")
    print(f"总用水 (WHW): {mean_absolute_error(y_test[:, 0], y_pred[:, 0]):.2f}")
    print(f"热水   (HTW): {mean_absolute_error(y_test[:, 1], y_pred[:, 1]):.2f}")
    print(f"冷水   (DWW): {mean_absolute_error(y_test[:, 2], y_pred[:, 2]):.2f}")

    joblib.dump(multi_xgb, '../checkpoints/water_multi_model.pkl')
    print("\nXGBoost水务模型已保存")

if __name__ == '__main__':
    train_gas_model()
    train_water_model_expert()