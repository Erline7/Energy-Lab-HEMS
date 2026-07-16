import pandas as pd
import os
from datetime import datetime, time
from typing import List, Optional

from sqlalchemy import (
    create_engine, String, Float, Integer, DateTime, Time, ForeignKey, Index
)
from sqlalchemy.orm import (
    DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "hems_data.db").replace("\\", "/")
DB_URI = f"sqlite:///{DB_PATH}"
engine = create_engine(DB_URI, echo=False, connect_args={'check_same_thread': False})
SessionLocal = sessionmaker(bind=engine)


class Base(DeclarativeBase):
    pass

class Appliance(Base):
    __tablename__ = "appliances"
    app_code: Mapped[str] = mapped_column(String(10), primary_key=True, comment="电器代号")
    app_name: Mapped[str] = mapped_column(String(50), nullable=False)
    nominal_power_w: Mapped[float] = mapped_column(Float, nullable=False)
    manual_pdf_names: Mapped[Optional[str]] = mapped_column(String(500), comment="说明书文件名")

    nilm_logs: Mapped[List["NilmInferredLog"]] = relationship(back_populates="appliance", cascade="all, delete-orphan")


class TelemetryMain(Base):
    __tablename__ = "telemetry_main"
    timestamp: Mapped[datetime] = mapped_column(DateTime, primary_key=True)
    whe_power: Mapped[float] = mapped_column(Float, nullable=False)
    temperature: Mapped[Optional[float]] = mapped_column(Float)
    humidity: Mapped[Optional[float]] = mapped_column(Float)

    # 归一化后的电器功率
    CWE_Power_Norm: Mapped[Optional[float]] = mapped_column(Float, default=0.0)
    HPE_Power_Norm: Mapped[Optional[float]] = mapped_column(Float, default=0.0)
    BME_Power_Norm: Mapped[Optional[float]] = mapped_column(Float, default=0.0)
    CDE_Power_Norm: Mapped[Optional[float]] = mapped_column(Float, default=0.0)
    WOE_Power_Norm: Mapped[Optional[float]] = mapped_column(Float, default=0.0)
    TVE_Power_Norm: Mapped[Optional[float]] = mapped_column(Float, default=0.0)
    FGE_Power_Norm: Mapped[Optional[float]] = mapped_column(Float, default=0.0)

    # 电器开关状态
    CWE_State: Mapped[Optional[int]] = mapped_column(Integer, default=0)
    HPE_State: Mapped[Optional[int]] = mapped_column(Integer, default=0)
    BME_State: Mapped[Optional[int]] = mapped_column(Integer, default=0)
    CDE_State: Mapped[Optional[int]] = mapped_column(Integer, default=0)
    WOE_State: Mapped[Optional[int]] = mapped_column(Integer, default=0)
    TVE_State: Mapped[Optional[int]] = mapped_column(Integer, default=0)
    FGE_State: Mapped[Optional[int]] = mapped_column(Integer, default=0)

    __table_args__ = (
        Index('idx_telemetry_timestamp', timestamp.desc()),
    )


class NilmInferredLog(Base):
    __tablename__ = "nilm_inferred_logs"
    timestamp: Mapped[datetime] = mapped_column(DateTime, primary_key=True)
    app_code: Mapped[str] = mapped_column(ForeignKey("appliances.app_code"), primary_key=True)
    current_state: Mapped[int] = mapped_column(Integer, nullable=False)
    inferred_power: Mapped[float] = mapped_column(Float, nullable=False)

    appliance: Mapped["Appliance"] = relationship(back_populates="nilm_logs")

    __table_args__ = (
        Index('idx_nilm_logs_app_time', app_code, timestamp.desc()),
    )


class WaterMainLog(Base):
    __tablename__ = "water_main_logs"
    timestamp: Mapped[datetime] = mapped_column(DateTime, primary_key=True)
    counter: Mapped[float] = mapped_column(Float, nullable=False)
    avg_rate: Mapped[float] = mapped_column(Float, nullable=False)
    inst_rate: Mapped[Optional[float]] = mapped_column(Float)


class WaterSubLog(Base):
    __tablename__ = "water_sub_logs"
    timestamp: Mapped[datetime] = mapped_column(DateTime, primary_key=True)
    app_code: Mapped[str] = mapped_column(String(10), primary_key=True)
    avg_rate: Mapped[float] = mapped_column(Float, nullable=False)


class NaturalGasLog(Base):
    __tablename__ = "natural_gas_logs"
    month_date: Mapped[datetime] = mapped_column(DateTime, primary_key=True)
    consumption_gj: Mapped[float] = mapped_column(Float, nullable=False)
    avg_outside_temp: Mapped[float] = mapped_column(Float, nullable=False)


# ---------- 数据导入函数 ----------
def init_core_data():
    """创建表并写入电器元数据"""
    print("正在创建数据库表...")
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    with SessionLocal() as session:
        appliances_data = [
            Appliance(app_code="CWE", app_name="洗衣机", nominal_power_w=500.0, manual_pdf_names="Manual_FTF2140.pdf"),
            Appliance(app_code="HPE", app_name="热泵/空调", nominal_power_w=1500.0,
                      manual_pdf_names="Manual_Heritage16.pdf, Manual_4a6h6036.pdf"),
            Appliance(app_code="BME", app_name="照明", nominal_power_w=300.0, manual_pdf_names="Manual_587144034.pdf"),
            Appliance(app_code="CDE", app_name="干衣机", nominal_power_w=3000.0, manual_pdf_names="Manual_FEQ1442.pdf"),
            Appliance(app_code="WOE", app_name="电烤箱", nominal_power_w=2500.0,
                      manual_pdf_names="Manual_hbl5045auc.pdf"),
            Appliance(app_code="TVE", app_name="电视/娱乐系统", nominal_power_w=200.0,
                      manual_pdf_names="Manual_NR1403.pdf, Manual_TCP42ST30.pdf"),
            Appliance(app_code="FGE", app_name="冰箱", nominal_power_w=150.0, manual_pdf_names="Manual_Amana.pdf"),
        ]
        session.add_all(appliances_data)
        session.commit()
        print("电器元数据写入完成")


def import_water_data():
    """导入水务历史数据"""
    print("正在导入水务数据...")
    datasets_dir = os.path.join(BASE_DIR, "datasets")

    # 主水表
    whw_path = os.path.join(datasets_dir, "Water_WHW.csv")
    if os.path.exists(whw_path):
        df_whw = pd.read_csv(whw_path)
        df_whw['timestamp'] = pd.to_datetime(df_whw['unix_ts'], unit='s')
        df_whw[['timestamp', 'counter', 'avg_rate', 'inst_rate']].to_sql('water_main_logs', engine, if_exists='append',
                                                                         index=False)
        print("主水表数据导入完成")
    else:
        print(f"警告：找不到 {whw_path}，跳过主水表导入")

    # 子水表
    for file_name, code in [("Water_DWW.csv", "DWW"), ("Water_HTW.csv", "HTW")]:
        path = os.path.join(datasets_dir, file_name)
        if os.path.exists(path):
            df_sub = pd.read_csv(path)
            df_sub['timestamp'] = pd.to_datetime(df_sub['unix_ts'], unit='s')
            df_sub['app_code'] = code
            df_sub[['timestamp', 'app_code', 'avg_rate']].to_sql('water_sub_logs', engine, if_exists='append',
                                                                 index=False)
            print(f"{code} 子水表数据导入完成")
        else:
            print(f"警告：找不到 {path}，跳过 {code} 导入")


def import_natural_gas_data():
    """导入燃气历史数据"""
    print("正在导入燃气数据...")
    path = os.path.join(BASE_DIR, "datasets", "NaturalGas_Monthly.csv")
    if os.path.exists(path):
        df_usage = pd.read_csv(path)
        df_usage['month_date'] = pd.to_datetime(df_usage['Month Year'], format='%b %Y')
        df_usage = df_usage.rename(columns={
            'Net Consumption (GJ)': 'consumption_gj',
            'Avg Outside Temp (℃)': 'avg_outside_temp'
        })
        df_usage[['month_date', 'consumption_gj', 'avg_outside_temp']].to_sql('natural_gas_logs', engine,
                                                                              if_exists='append', index=False)
        print("燃气数据导入完成")
    else:
        print(f"警告：找不到燃气数据文件 {path}")


def seed_telemetry_and_nilm():
    """从CSV导入电力遥测数据和NILM分解日志（原 seed_data.py 的核心功能）"""
    csv_path = os.path.join(BASE_DIR, "datasets", "nilm_inferred_states.csv")
    weather_path = os.path.join(BASE_DIR, "datasets", "Climate_HourlyWeather.csv")
    target_appliances = ['HPE', 'BME', 'CDE', 'CWE', 'WOE', 'TVE', 'FGE']

    if not os.path.exists(csv_path):
        print(f"错误：找不到电力数据文件 {csv_path}")
        return

    session = SessionLocal()

    # 清空已有遥测和NILM日志（避免重复）
    try:
        session.query(NilmInferredLog).delete()
        session.query(TelemetryMain).delete()
        session.commit()
        print("已清空 telemetry_main 和 nilm_inferred_logs 旧数据")
    except Exception as e:
        session.rollback()
        print(f"清空数据时出错: {e}")

    df = pd.read_csv(csv_path)

    # 识别时间列
    potential_time_cols = ['Time', 'time', 'timestamp', df.columns[0]]
    time_col = next((col for col in potential_time_cols if col in df.columns), None)
    if time_col is None:
        print("错误：CSV中未找到时间列")
        session.close()
        return
    df = df.rename(columns={time_col: 'Time'})
    df['Time'] = pd.to_datetime(df['Time'])
    if df['Time'].dt.tz is not None:
        df['Time'] = df['Time'].dt.tz_localize(None)
    df = df.dropna(subset=['Time'])

    # 处理气象数据（15分钟线性插值并合并）
    if os.path.exists(weather_path):
        print("正在读取气象数据并执行15分钟线性插值...")
        df_w = pd.read_csv(weather_path)
        df_w = df_w[['Date/Time', 'Temp (C)', 'Rel Hum (%)']].copy()
        df_w = df_w.rename(columns={'Date/Time': 'Time', 'Temp (C)': 'temp', 'Rel Hum (%)': 'hum'})
        df_w['Time'] = pd.to_datetime(df_w['Time'])
        if df_w['Time'].dt.tz is not None:
            df_w['Time'] = df_w['Time'].dt.tz_localize(None)
        df_w = df_w.dropna(subset=['Time']).drop_duplicates(subset=['Time'], keep='first')
        df_w = df_w.set_index('Time').resample('15min').interpolate(method='linear')
        df = pd.merge(df, df_w, on='Time', how='left')
        df['temp'] = df['temp'].ffill().bfill().fillna(20.0)
        df['hum'] = df['hum'].ffill().bfill().fillna(50.0)
        print("气象数据合并完成")
    else:
        print("警告：未找到气象数据文件，使用默认温湿度")
        df['temp'] = 20.0
        df['hum'] = 50.0

    # 写入 TelemetryMain
    main_p_col = 'WHE_True' if 'WHE_True' in df.columns else 'WHE'
    main_records = []
    for _, row in df.iterrows():
        record = {
            'timestamp': row['Time'],
            'whe_power': row[main_p_col],
            'temperature': row['temp'],
            'humidity': row['hum'],
        }
        for app in target_appliances:
            power_col = f"{app}_Power_Norm"
            state_col = f"{app}_State"
            record[f'{app}_Power_Norm'] = row.get(power_col, 0.0)
            record[f'{app}_State'] = int(row.get(state_col, 0))
        main_records.append(record)

    session.bulk_insert_mappings(TelemetryMain, main_records)
    session.commit()
    print(f"成功写入 {len(main_records)} 条遥测主数据")

    # 写入 NilmInferredLog
    for app in target_appliances:
        power_col = f"{app}_Power_Norm"
        state_col = f"{app}_State"
        if power_col in df.columns and state_col in df.columns:
            app_df = df[['Time', power_col, state_col]].copy()
            app_df['inferred_power'] = app_df[power_col] * 6900.0
            app_df['current_state'] = app_df[state_col].astype(int)
            app_df['app_code'] = app
            app_df = app_df.rename(columns={'Time': 'timestamp'})
            records = app_df[['timestamp', 'app_code', 'current_state', 'inferred_power']].to_dict(orient='records')
            session.bulk_insert_mappings(NilmInferredLog, records)
            session.commit()
            print(f"成功写入 {len(records):,} 条 {app} 的NILM日志")
        else:
            print(f"跳过 {app}：缺少列 {power_col} 或 {state_col}")

    session.close()
    print("电力与NILM数据同步完成")


# ---------- 主入口 ----------
if __name__ == "__main__":
    print("开始初始化数据库")
    init_core_data()           # 创建表 + 电器元数据
    seed_telemetry_and_nilm()  # 导入遥测和NILM数据
    import_water_data()        # 导入水务数据
    import_natural_gas_data()  # 导入燃气数据
    print("\n所有数据导入成功")