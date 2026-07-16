import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from forecaster_model import Energy_dataset, WHEModel

class HybridLoss(nn.Module):
    def __init__(self, max_power, quantiles=[0.1, 0.5, 0.9], mse_weight=0.5, q_weights=[1.5, 1.0, 2.0]):
        super().__init__()
        self.quantiles  = quantiles
        self.mse_weight = mse_weight
        self.q_weights  = q_weights
        self.mse        = nn.MSELoss()

        # 使用动态 max_power 计算阈值（归一化尺度）
        self.thresh_medium  = 884.0 / max_power
        self.thresh_high    = 1140.0 / max_power
        self.thresh_extreme = 1700.0 / max_power
        
    def forward(self, preds, target):
        # preds: (B, T, 3)   target: (B, T, 1)
        # ---- 1. 分位数损失 (Quantile Loss) ----
        target_exp = target.expand_as(preds)  # (B, T, 3)
        q_loss = 0.0
        for i, q in enumerate(self.quantiles):
            err = target_exp[..., i] - preds[..., i]
            q_loss += self.q_weights[i] * torch.max((q - 1) * err, q * err).mean()

        # ---- 2. MSE 损失 (针对 P50) ----
        mse_loss = self.mse(preds[..., 1:2], target)  # 形状 (B, T, 1) vs (B, T, 1)

        # ---- 3. 峰值惩罚损失 (Peak Penalty) ----
        # 直接使用三维张量，避免 squeeze 导致维度错乱
        err_p50 = F.l1_loss(preds[..., 1:2], target, reduction='none')  # (B, T, 1)

        mask_medium  = (target > self.thresh_medium).float()
        mask_high    = (target > self.thresh_high).float()
        mask_extreme = (target > self.thresh_extreme).float()

        peak_loss = (
                err_p50 * mask_medium * 0.5
                + err_p50 * mask_high * 0.5  
                + err_p50 * mask_extreme * 0.5  
        ).mean()

        return (1 - self.mse_weight) * q_loss + self.mse_weight * mse_loss + 0.2 * peak_loss

def build_lr_lambda(warmup_epochs: int, total_epochs: int):
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda

def augment(x: torch.Tensor, noise_std: float = 0.015, dropout_p: float = 0.04) -> torch.Tensor:
    x = x + torch.randn_like(x) * noise_std
    # step_mask 形状为 (B, T, 1)，广播到所有特征维度
    step_mask = (torch.rand(x.shape[0], x.shape[1], 1, device=x.device) > dropout_p).float()
    return x * step_mask

def train_forecaster():
    CSV_PATH    = '../datasets/nilm_inferred_states.csv'
    HISTORY_LEN = 96
    FUTURE_LEN  = 96
    BATCH_SIZE  = 32
    EPOCHS      = 50
    LR          = 1e-4
    WARMUP      = 5
    DEVICE      = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    USE_AMP     = False
    SAVE_PATH   = '../checkpoints/best_day_ahead_forecaster.pth'

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    print("加载数据集...")
    full_dataset = Energy_dataset(
        CSV_PATH,
        weather_csv_path='../datasets/Climate_HourlyWeather.csv',
        history_len=HISTORY_LEN,
        future_len=FUTURE_LEN,
        mode='refined',
        max_power=None
    )
    dynamic_max_power = full_dataset.max_power
    print(f"从数据中检测到最大功率: {dynamic_max_power:.2f} W")

    n_nilm = len(full_dataset.nilm_cols_used)

    total_samples = len(full_dataset)
    val_start     = int(total_samples * 0.80)
    
    train_indices = list(range(val_start))
    val_indices   = list(range(val_start, total_samples))

    train_ds = Subset(full_dataset, train_indices)
    val_ds   = Subset(full_dataset, val_indices)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=False   
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    print(f"总样本数 {total_samples:,} | 训练集: {len(train_ds):,} | 验证集: {len(val_ds):,}")
    print(f"验证集区间索引: [{val_start:,}, {total_samples:,})")

    print("初始化 WHE 预测模型...")
    model = WHEModel(
        input_dim=full_dataset.input_dim,
        n_nilm_features=n_nilm,
        cnn_filters=16,
        hidden_dim=256
    ).to(DEVICE)

    optimizer  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler  = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=build_lr_lambda(warmup_epochs=WARMUP, total_epochs=EPOCHS)
    )

    amp_scaler = torch.amp.GradScaler('cuda') if USE_AMP and DEVICE != 'cpu' else None

    criterion_train = HybridLoss(
        max_power=dynamic_max_power,
        quantiles=[0.1, 0.5, 0.9],
        mse_weight=0.5,
        q_weights=[1.5, 1.0, 2.0]
    )
    criterion_eval = nn.L1Loss()

    best_val_loss      = float('inf')
    early_stop_counter = 0

    print("开始训练...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_total = 0.0

        for x_hist, y_power in train_loader:
            x_hist, y_power = x_hist.to(DEVICE), y_power.to(DEVICE)
            x_hist = augment(x_hist)
            optimizer.zero_grad()
            
            if USE_AMP and DEVICE != 'cpu':
                with torch.amp.autocast('cuda'):
                    pred_power = model(x_hist)
                    loss_power = criterion_train(pred_power, y_power)
                loss = 2.0 * loss_power
                amp_scaler.scale(loss).backward()
                amp_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                amp_scaler.step(optimizer)
                amp_scaler.update()
            else:
                pred_power = model(x_hist)
                loss_power = criterion_train(pred_power, y_power)
                loss = 2.0 * loss_power
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            train_loss_total += loss.item()

        scheduler.step()

        # ---------- 验证 ----------
        model.eval()
        val_loss_total     = 0.0
        val_p90_loss_total = 0.0
        total_in_interval  = 0
        total_points       = 0

        with torch.no_grad():
            for x_hist, y_power in val_loader:
                x_hist, y_power = x_hist.to(DEVICE), y_power.to(DEVICE)

                if USE_AMP and DEVICE != 'cpu':
                    with torch.amp.autocast('cuda'):
                        pred_power = model(x_hist)
                else:
                    pred_power = model(x_hist)

                loss_p50   = criterion_eval(pred_power[..., 1:2], y_power)
                loss_p90   = criterion_eval(pred_power[..., 2:3], y_power)

                val_loss_total     += loss_p50.item()
                val_p90_loss_total += loss_p90.item()

                p10 = pred_power[..., 0:1]
                p90 = pred_power[..., 2:3]
                in_interval        = ((y_power >= p10) & (y_power <= p90)).float()
                total_in_interval += in_interval.sum().item()
                total_points      += in_interval.numel()

        avg_val_loss       = val_loss_total    / len(val_loader)
        avg_p90_loss       = val_p90_loss_total / len(val_loader)
        real_mae_watts     = avg_val_loss  * dynamic_max_power
        real_p90_mae_watts = avg_p90_loss  * dynamic_max_power
        coverage           = total_in_interval / total_points * 100
        current_lr         = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"训练损失: {train_loss_total / len(train_loader):.4f} | "
              f"验证 MAE(P50): {real_mae_watts:.1f}W | "
              f"验证 MAE(P90): {real_p90_mae_watts:.1f}W | "
              f"覆盖率: {coverage:.1f}% | "
              f"学习率: {current_lr:.2e}")

        if avg_val_loss < best_val_loss:
            best_val_loss      = avg_val_loss
            early_stop_counter = 0
            checkpoint = {
                'model_state': model.state_dict(),
                'max_power': dynamic_max_power,
                'in_channels': full_dataset.input_dim,
                'history_len': HISTORY_LEN,
                'future_len': FUTURE_LEN,
                'n_nilm_features': n_nilm
            }
            torch.save(checkpoint, SAVE_PATH)
            print(f"保存最佳模型（验证 MAE P50: {real_mae_watts:.1f}W | 覆盖率: {coverage:.1f}%）")
        else:
            early_stop_counter += 1

        if early_stop_counter >= 10:
            print("触发早停，提前结束训练")
            break

if __name__ == '__main__':
    train_forecaster()