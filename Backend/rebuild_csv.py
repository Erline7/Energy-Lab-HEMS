import os
os.environ["OMP_NUM_THREADS"] = "5"
import numpy as np
from NILM.event_detection import load, clean, features, fit_gmm

TARGET_APPLIANCES = ['HPE', 'BME', 'CDE', 'CWE', 'WOE', 'TVE', 'FGE']
DATA_DIR = 'datasets/'

def main():
    print("\n=== 开始重构家庭能源预测特征集 ===")
    circuits_to_read = ['WHE', 'RSE', 'GRE'] + TARGET_APPLIANCES

    if not os.path.exists(DATA_DIR):
        print(f"目录不存在: {DATA_DIR}")
        return
    
    missing_files = []
    for circuit in circuits_to_read:
        fpath = os.path.join(DATA_DIR, f'Electricity_{circuit}.csv')
        if not os.path.exists(fpath):
            missing_files.append(fpath)

    if missing_files:
        print(f"缺失{len(missing_files)}个文件：")
        for f in missing_files[:5]:
            print(f"   - {f}")
        return
        
    try:
        df = load(DATA_DIR, circuits_to_read)
        if df is None or df.empty:
            print("数据为空")
            return
        print(f"读取成功，数据形状为{df.shape}")
    except Exception as e:
        print(f"读取数据时出错: {e}")
        return

    print("\n1. 清洗与对齐数据")
    df = clean(df)
    
    # 【修复点 1】和 NILM 完全对齐的净负荷计算逻辑
    df['WHE_True'] = df['WHE'] - (df['RSE'].fillna(0) + df['GRE'].fillna(0))
    df['WHE_True'] = df['WHE_True'].clip(lower=0.0)
    
    # 【修复点 2】动态提取最大功率，消除硬编码 6900.0
    global_max_power = float(df['WHE_True'].max())
    print(f"   提取到全局总功率最大值: {global_max_power:.2f} W")

    print("\n2. 生成GMM状态标签")
    train_split = int(len(df) * 0.7)
    train_raw_apps = df[TARGET_APPLIANCES].values[:train_split]
    all_raw_apps = df[TARGET_APPLIANCES].values

    if len(train_raw_apps) == 0 or len(all_raw_apps) == 0:
        print("训练数据为空！")
        return

    try:
        # 这里传入 appliance_name=TARGET_APPLIANCES，和 nilm_train 内部对齐
        all_labels, n_states_per_app = fit_gmm(
            train=train_raw_apps,
            all_data=all_raw_apps,
            appliance_name=TARGET_APPLIANCES,
            n_range=(2, 6)
        )
        print(f"   各电器状态数: {dict(zip(TARGET_APPLIANCES, n_states_per_app))}")
    except Exception as e:
        print(f"GMM聚类出错: {e}")
        return

    print("\n3. 注入特征与归一化")
    for i, app in enumerate(TARGET_APPLIANCES):
        df[f'{app}_State'] = all_labels[:, i]
        # 使用动态提取的最大功率进行归一化，与 Forecaster 保持联动
        df[f'{app}_Power_Norm'] = df[app] / global_max_power

    # 【修复点 3】传入 'HPE' 以欺骗 features 函数，确保全局生成 cos_month
    df = features(df, appliance_name='HPE')

    final_cols = ['WHE_True']
    for app in TARGET_APPLIANCES:
        final_cols.append(f'{app}_Power_Norm')
        final_cols.append(f'{app}_State')

    # 将预测模型需要的时序特征统统保留
    final_cols += ['sin_hour', 'cos_hour', 'is_weekend', 'sin_month', 'cos_month']

    missing_cols = [col for col in final_cols if col not in df.columns]
    if missing_cols:
        print(f"警告：以下列未能生成，将被剔除: {missing_cols}")
        final_cols = [col for col in final_cols if col in df.columns]

    final_df = df[final_cols]

    save_path = 'datasets/nilm_inferred_states.csv'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    final_df.to_csv(save_path, index=True)

    print(f"\n=== 构建完成 ===")
    print(f"文件已保存至: {save_path}")
    print(f"最终数据形状: {final_df.shape}")

if __name__ == '__main__':
    main()