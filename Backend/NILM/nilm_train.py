import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from NILM.event_detection import load, clean, split_data, normalize, features, fit_gmm

TARGET_APPLIANCES = ['HPE', 'BME', 'CDE', 'WOE', 'TVE', 'FGE', 'CWE']

class TCNResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, padding, dilation):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2 
        self.conv = nn.Conv1d(input_dim, output_dim, kernel_size, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU(inplace=True)
        self.residual = nn.Conv1d(input_dim, output_dim, kernel_size=1) if input_dim != output_dim else nn.Identity()
    def forward(self, x):
        res = self.residual(x)
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out + res)
        return out

class TCN_BiGRU(nn.Module):
    def __init__(self, window_size, n_appliances, n_states_per_app, input_dim = 5, hidden_dim = 64):
        super().__init__()
        if isinstance(n_states_per_app, int):
            n_states_per_app = [n_states_per_app]
        self.ns = n_states_per_app
        self.tcn1 = TCNResidualBlock(input_dim, 16, kernel_size=5, padding=0,dilation=1)
        self.tcn2 = TCNResidualBlock(16, 32, kernel_size=5, padding=0,dilation=2)
        self.tcn3 = TCNResidualBlock(32, 64, kernel_size=5, padding=0,dilation=4)

        self.bigru = nn.GRU(input_size=64, hidden_size=hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(nn.Linear(hidden_dim * 2, 64), nn.ReLU(), nn.Dropout(0.3))
        self.power_regressor = nn.Linear(64, n_appliances)
        self.state_classifiers = nn.ModuleList([nn.Linear(64, i) for i in self.ns])
    def forward(self, x):
        x_tcn = x.permute(0, 2, 1)
        out = self.tcn1(x_tcn)
        out = self.tcn2(out)
        out = self.tcn3(out)
        x_gru = out.permute(0, 2, 1)
        gru_out, _ = self.bigru(x_gru)
        center_idx = gru_out.shape[1] // 2
        center_feat = gru_out[:, center_idx, :]
        feat = self.fc(center_feat)
        power_pred = self.power_regressor(feat)
        logits_list = [classifier(feat) for classifier in self.state_classifiers]
        return power_pred, logits_list


class NILMDataset(Dataset):
    def __init__(self, features, appliances_data, states_labels, window_size=59):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.appliances_data = torch.as_tensor(appliances_data, dtype=torch.float32)
        self.states_labels = torch.as_tensor(states_labels, dtype=torch.long)

        self.window_size = window_size
        self.pad_len = window_size // 2
        self.n_samples = len(features) - window_size + 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        x_window = self.features[index: index + self.window_size]
        center_index = index + self.pad_len
        y_power = self.appliances_data[center_index]
        y_state = self.states_labels[center_index]
        return x_window, y_power, y_state


def prepare_data(data_dir, appliances, window_size=199, batch_size=64, n_range=(2, 6)):
    cache_file = os.path.join(data_dir, 'cache', f'cache_data_{appliances}.pt')
    USE_CACHE = True

    if os.path.exists(cache_file) and USE_CACHE:
        print(f'找到缓存文件 {cache_file}，直接加载')
        data_dict = torch.load(cache_file, weights_only=False)
        train_ds = NILMDataset(data_dict['train_x'], data_dict['train_y'], data_dict['train_s'], window_size)
        val_ds   = NILMDataset(data_dict['val_x'],   data_dict['val_y'],   data_dict['val_s'],   window_size)
        test_ds  = NILMDataset(data_dict['test_x'],  data_dict['test_y'],  data_dict['test_s'],  window_size)
        meta = data_dict['meta']

    else:
        print("未找到缓存，开始处理数据")

        main_appliances = ['WHE', 'RSE', 'GRE', appliances]
        df = load(data_dir, main_appliances)
        df = clean(df)
        df['WHE'] = df['WHE'] - (df['RSE'] + df['GRE'])
        df['WHE'] = df['WHE'].clip(lower=0.0)
        df = df.drop(columns=['RSE', 'GRE'])
        df = features(df, appliance_name=appliances)
        raw = df.values.astype(np.float32)

        raw_diff = np.diff(raw[:, 0:1], axis=0, prepend=raw[0:1, 0:1])
        raw_ext = np.concatenate([raw[:, 0:2], raw_diff, raw[:, 2:]], axis=1)
        T_total = len(raw_ext)

        train_raw, val_raw, test_raw, i1, i2 = split_data(raw_ext)
        N_POWER_COLS = 3
        scaler, train_norm, val_norm, test_norm = normalize(
            train_raw, val_raw, test_raw, n_power_cols=N_POWER_COLS
        )

        app_names     = [appliances]
        app_train_raw = train_raw[:, 1:2]
        train_labels, n_states = fit_gmm(
            train=app_train_raw,
            all_data=app_train_raw,
            appliance_name=app_names,
            n_range=n_range
        )
        print(f"{appliances} 有 {n_states} 种状态")

        app_all_raw = raw_ext[:, 1:2]
        all_labels_full, _ = fit_gmm(
            train=app_train_raw,
            all_data=app_all_raw,
            appliance_name=app_names,
            n_range=n_range
        )

        def build_features(norm_data, label_start, label_end):
            main_power = norm_data[:, 0:1]
            diff_power = norm_data[:, 2:3]
            time_feats = norm_data[:, 3:]
            x_in = np.concatenate([main_power, diff_power, time_feats], axis=1)
            appliances_data = norm_data[:, 1:2]
            state_labels = all_labels_full[label_start:label_end]
            return x_in, appliances_data, state_labels

        train_x, train_y, train_s = build_features(train_norm, 0,  i1)
        val_x,   val_y,   val_s   = build_features(val_norm,  i1,  i2)
        test_x,  test_y,  test_s  = build_features(test_norm, i2,  T_total)

        meta = {'scaler': scaler, 'ns': n_states, 'n_appliances': 1,
                'n_power_cols': N_POWER_COLS}
        data_dict = {
            'train_x': train_x, 'train_y': train_y, 'train_s': train_s,
            'val_x':   val_x,   'val_y':   val_y,   'val_s':   val_s,
            'test_x':  test_x,  'test_y':  test_y,  'test_s':  test_s,
            'meta': meta
        }
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        torch.save(data_dict, cache_file)
        print(f"缓存已保存到: {cache_file}")

        train_ds = NILMDataset(train_x, train_y, train_s, window_size)
        val_ds   = NILMDataset(val_x,   val_y,   val_s,   window_size)
        test_ds  = NILMDataset(test_x,  test_y,  test_s,  window_size)

    loader_settings = dict(
        num_workers=0,
        pin_memory=True,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True, **loader_settings)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **loader_settings)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **loader_settings)

    return train_loader, val_loader, test_loader, meta


def nilm_loss(pred_power, true_power, pred_states_list, true_states, power_weight=2.0, state_weight=1.0):
    power_loss = nn.MSELoss()(pred_power, true_power)
    n_classes = pred_states_list[0].shape[1]
    device = pred_power.device
    weights = torch.ones(n_classes, device=device)
    weights[0] = 0.2  
    safe_states = torch.clamp(true_states[:, 0], min=0, max=n_classes - 1)
    state_loss = nn.CrossEntropyLoss(weight=weights)(pred_states_list[0], safe_states)
    total_loss = power_weight * power_loss + state_weight * state_loss
    return total_loss, power_loss, state_loss

@torch.no_grad()
def evaluate_nilm(model, data_loader, device, scaler, return_arrays=False):
    model.eval()
    all_predictions = []
    all_true_values = []
    all_state_logits = []
    total_state_loss = 0.0

    for windows, true_power, true_states in data_loader:
        windows     = windows.to(device)
        true_power  = true_power.to(device)
        true_states = true_states.to(device)
        pred_power, pred_states = model(windows)
        _, _, state_loss = nilm_loss(pred_power, true_power, pred_states, true_states)
        total_state_loss += state_loss.item()
        all_predictions.append(pred_power.cpu().numpy())
        all_true_values.append(true_power.cpu().numpy())
        all_state_logits.append(pred_states[0].cpu().numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_true_values = np.concatenate(all_true_values, axis=0)
    all_state_logits = np.concatenate(all_state_logits, axis=0)

    pred_states_classes = np.argmax(all_state_logits, axis=1)
    def inverse_appliance(normalized_col):
        dummy = np.zeros((len(normalized_col), 3))
        dummy[:, 1] = normalized_col[:, 0]
        return scaler.inverse_transform(dummy)[:, 1:2]

    pred_power_watts = inverse_appliance(all_predictions)
    true_power_watts = inverse_appliance(all_true_values)
    pred_power_watts[pred_states_classes == 0] = 0.0
    pred_power_watts = np.clip(pred_power_watts, a_min=0.0, a_max=None)

    mae = np.mean(np.abs(pred_power_watts - true_power_watts))
    mse = np.mean(np.square(pred_power_watts - true_power_watts))
    rmse = np.sqrt(mse)

    avg_state_loss = total_state_loss / len(data_loader)

    if return_arrays:
        return mae, rmse, avg_state_loss, true_power_watts, pred_power_watts
    return mae, rmse, avg_state_loss


def train_single_appliance(appliances):
    DATA_DIR  = '../datasets'
    WINDOW_SIZE = 99
    BATCH_SIZE  = 64
    EPOCHS      = 50
    LR          = 1e-3
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    SAVE_PATH = f'../checkpoints/best_nilm_{appliances}.pth'
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    train_loader, val_loader, test_loader, meta = prepare_data(
        DATA_DIR, appliances=appliances,
        window_size=WINDOW_SIZE, batch_size=BATCH_SIZE
    )
    n_states = meta['ns']
    scaler   = meta['scaler']

    channels = 6 if appliances == 'HPE' else 5

    # 替换为新的轻量级模型 TCN_BiGRU
    model = TCN_BiGRU(
        window_size=WINDOW_SIZE,
        n_appliances=1,
        n_states_per_app=n_states,
        input_dim=channels,
        hidden_dim=64
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    amp_scaler = torch.amp.GradScaler('cuda') if DEVICE == 'cuda:0' else None

    best_val_rmse = float('inf')
    early_stop_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for x_win, y_power, y_state in train_loader:
            x_win, y_power, y_state = (
                x_win.to(DEVICE), y_power.to(DEVICE), y_state.to(DEVICE)
            )
            optimizer.zero_grad()

            if amp_scaler is not None:
                with torch.amp.autocast('cuda'):
                    p_pred, s_logits = model(x_win)
                    loss, _, _ = nilm_loss(p_pred, y_power, s_logits, y_state)
                amp_scaler.scale(loss).backward()
                amp_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                amp_scaler.step(optimizer)
                amp_scaler.update()
            else:
                p_pred, s_logits = model(x_win)
                loss, _, _ = nilm_loss(p_pred, y_power, s_logits, y_state)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            train_loss += loss.item()

        val_mae, val_rmse, _ = evaluate_nilm(model, val_loader, DEVICE, scaler)

        scheduler.step(val_rmse)

        print(
            f"Epoch {epoch}/{EPOCHS} | Loss {train_loss / len(train_loader):.4f} | {appliances} Val MAE: {val_mae:.2f} W | Val RMSE: {val_rmse:.2f} W")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            early_stop_counter = 0
            torch.save({
                'model_state': model.state_dict(),
                'ns': n_states,
                'window_size': WINDOW_SIZE,
                'n_appliances': 1,
                'scaler': scaler 
            }, SAVE_PATH)
            print(f"保存最佳模型 (RMSE: {best_val_rmse:.2f}W, MAE: {val_mae:.2f}W)")
        else:
            early_stop_counter += 1
            print(f"连续 {early_stop_counter} 轮 RMSE 没进步")
            if early_stop_counter >= 10:
                print(f"\n连续 10 轮 RMSE 没进步，提前结束训练！")
                break

    print(f"\n{appliances} 训练完成")
    ckpt = torch.load(SAVE_PATH, weights_only=False)
    model.load_state_dict(ckpt['model_state'])

    test_mae, test_rmse, _, true_arr, pred_arr = evaluate_nilm(
        model, test_loader, DEVICE, scaler, return_arrays=True
    )
    print(f'{appliances} 最终测试集 -> MAE: {test_mae:.2f} W | RMSE: {test_rmse:.2f} W')
    print("=" * 60)

if __name__ == '__main__':
    print("开始训练所有电器")
    for app in TARGET_APPLIANCES:
        train_single_appliance(app)
    print("所有电器的模型已训练完成！")