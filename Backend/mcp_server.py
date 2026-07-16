import os
import json
import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
import pickle
import threading
import traceback
import logging
import torch
import pandas as pd
from rank_bm25 import BM25Okapi
from FlagEmbedding import FlagReranker
import numpy as np
import joblib
from datetime import timedelta
import warnings
import fitz
import cv2
import re
from dotenv import load_dotenv
load_dotenv()
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from functools import lru_cache
from langchain_chroma import Chroma
from sqlalchemy import create_engine, text
from langchain_core.tools import tool
from typing import Dict, Any
from mcp.server.fastmcp import FastMCP
from rapidocr_onnxruntime import RapidOCR
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
fitz.TOOLS.mupdf_display_errors(False)
warnings.filterwarnings('ignore')

vectorstore_lock = threading.Lock()
progress_lock = threading.Lock()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DB_URI = "sqlite:///hems_data.db"
engine = create_engine(DB_URI, connect_args={'check_same_thread': False})

APPLIANCE_MAP = {
    "热泵": "HPE", "空调": "HPE", "干衣机": "CDE", "烘干机": "CDE",
    "洗衣机": "CWE", "冰箱": "FGE", "烤箱": "WOE", "电视": "TVE", "照明": "BME"
}
REVERSE_APPLIANCE_MAP = {
    "HPE": "热泵", "CDE": "干衣机", "CWE": "洗衣机",
    "FGE": "冰箱", "WOE": "烤箱", "TVE": "电视", "BME": "照明"
}
APPLIANCES_LIST = ['HPE', 'BME', 'CDE', 'WOE', 'TVE']

MODELS: Dict[str, Any] = {
    'forecaster': None,
    'forecaster_meta': {},
    'nilm': {},
    'water_gas': {'gas': None, 'water': None},
    'vectorstore': None,
    'ocr': None,
    'translator_chain': None,
    'bm25_index': None,          
    'bm25_docs': None,           
    'reranker': None,          
}

mcp = FastMCP("EnergyLab")


def initialize_all_resources(mode="all"):
    """初始化并预加载 AI 模型"""
    global MODELS
    logger.info(f"正在启动 AI 资源初始化 (模式: {mode})")

    if MODELS['forecaster'] is None:
        try:
            from Forecaster.forecaster_model import WHEModel
            ckpt_path = os.path.join(BASE_DIR, 'checkpoints/best_day_ahead_forecaster.pth')
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            MODELS['forecaster_meta'] = ckpt
            MODELS['forecaster_meta']['max_power'] = ckpt.get('max_power', 6900.0)

            n_nilm = ckpt.get('n_nilm_features', 7)

            model = WHEModel(
                input_dim=ckpt.get('in_channels', 21),
                n_nilm_features=n_nilm,
                cnn_filters=16,
                hidden_dim=256,
                future_len=96
            ).to(DEVICE)
            model.load_state_dict(ckpt['model_state'])
            model.eval()
            MODELS['forecaster'] = model
            logger.info("负荷预测模型加载完成")
        except Exception as e:
            logger.info(f"负荷预测模型加载失败: {e}")
    if MODELS['water_gas']['gas'] is None:
        try:
            gas_path = os.path.join(BASE_DIR, 'checkpoints/gas_model.pkl')
            water_path = os.path.join(BASE_DIR, 'checkpoints/water_multi_model.pkl')
            MODELS['water_gas']['gas'] = joblib.load(gas_path)
            MODELS['water_gas']['water'] = joblib.load(water_path)
            logger.info("水气系统专家模型已就绪")
        except Exception as e:
            logger.info(f"水气专家模型缺失: {e}")
    if mode in ["all", "tools"]:
        if MODELS['vectorstore'] is None:
            try:
                emb_model_path = os.path.join(BASE_DIR, "local_models/BAAI/bge-base-zh-v1.5") 
                persist_dir = os.path.join(BASE_DIR, "chroma_db_appliances")
                embed = HuggingFaceEmbeddings(model_name=emb_model_path, model_kwargs={'device': DEVICE})
                MODELS['vectorstore'] = Chroma(persist_directory=persist_dir, embedding_function=embed)
                logger.info("说明书知识库已就绪")
            except Exception as e:
                logger.info(f"向量库初始化失败: {e}")

        if MODELS['ocr'] is None:
            try:
                MODELS['ocr'] = RapidOCR()
                logger.info("RapidOCR 视觉引擎已就绪")
            except Exception as e:
                logger.info(f"OCR 初始化失败: {e}")

        if MODELS['translator_chain'] is None:
            try:
                api_key = os.getenv("DASHSCOPE_API_KEY")
                if not api_key:
                    logger.warning("环境变量中未找到 DASHSCOPE_API_KEY！")
                llm_trans = ChatOpenAI(
                    model="qwen-plus",
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                    api_key=api_key,
                    temperature=0.0,     
                )
                prompt_tpl = ChatPromptTemplate.from_messages([
                    ("system", """你是一个极其严谨的工业级技术文档清洗与翻译专家，精通家电说明书的排版自愈。
                        你当前接收到的输入，是从英文说明书 OCR 或 PDF 中提取的原始文本。

                        请在翻译的同时，严格执行以下【布局清洗与重组规范】：
                        1. 【句子合并与自愈】：识别并重新组装因提取或断栏导致的断行、断句。还原正确的横向通排阅读语义。
                        2. 【页眉页脚降噪】：识别并彻底剔除无意义的页码、页脚标记、文档版本号、图表引线乱码（例如 "Fig 1-2"、"Page 24 of 50"、"......"）。
                        3. 【零无关字符原则】：你的输出必须是 100% 纯净的技术文档内容。严禁包含任何前言（如“好的”、“以下是为您整理的”）、后记、注释、表情包或交互式对话。
                        4. 【Markdown 优雅重构】：将翻译后的中文，智能重构为排版高亮、可读性强的 Markdown。适当使用 `###` 标题、`-` 符号列表以及 `|` 表格。"""),
                                    ("human", """请对以下提取出的文本内容进行【布局清洗、逻辑重组并严谨翻译为中文 Markdown】。

                        【非常重要的边界规则】：
                        - 乱码与型号豁免：遇到 OCR 乱码或专用电器型号（如 HBL5045AUC、Amana）时，请直接保持英文原样，千万不要强行翻译或解释。
                        - 零思考泄露原则：你的输出必须直接是中文 Markdown 内容，严禁包含任何形如 `<think>`、`思考：` 或者是推理心智过程的标签和字符。

                        待处理原始文本：
                        {text}""")
                ])
                MODELS['translator_chain'] = prompt_tpl | llm_trans | StrOutputParser()
                logger.info("模型已就绪")
            except Exception as e:
                logger.info(f"模型加载失败: {e}")

    
        if MODELS['bm25_index'] is None and MODELS['vectorstore'] is not None:
            try:
                logger.info("正在构建 BM25 关键词索引（这可能需要几秒钟，取决于说明书页数）...")
                all_data = MODELS['vectorstore']._collection.get()
                if all_data and len(all_data['documents']) > 0:
                    tokenized_docs = [doc.split() for doc in all_data['documents']]
                    MODELS['bm25_index'] = BM25Okapi(tokenized_docs)
                    MODELS['bm25_docs'] = all_data['documents']  # 缓存原文，用于后续索引重建
                    logger.info(f"BM25 索引构建完成，共 {len(MODELS['bm25_docs'])} 个块")
                else:
                    logger.warning("向量库为空，跳过 BM25 构建")
            except Exception as e:
                logger.error(f"BM25 构建失败（将降级为纯向量检索）: {e}")
                MODELS['bm25_index'] = None

        if MODELS['reranker'] is None:
            try:
                logger.info("正在加载 BGE Reranker（约 1.3GB，首次加载稍慢）...")
                MODELS['reranker'] = FlagReranker('BAAI/bge-reranker-base', use_fp16=True)
                logger.info("BGE Reranker 加载成功！")
            except Exception as e:
                logger.warning(f"Reranker 加载失败（将跳过精排）: {e}")
                MODELS['reranker'] = None

def get_nilm_model(app_name):
    if app_name not in MODELS['nilm']:
        from NILM.nilm_train import TCN_BiGRU
        try:
            ckpt = torch.load(f'checkpoints/best_nilm_{app_name}.pth', map_location=DEVICE, weights_only=False)
            scaler_path = f'checkpoints/nilm_scaler_{app_name}.pkl'
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
            else:
                scaler = None
                print(f"未找到 scaler 文件 {scaler_path}，将使用默认归一化")

          
            input_dim = 6 if app_name == 'HPE' else 5
            model = TCN_BiGRU(
                window_size=ckpt.get('window_size', 99),
                n_appliances=1,
                n_states_per_app=ckpt['ns'],
                input_dim=input_dim,
                hidden_dim=64
            ).to(DEVICE)
            model.load_state_dict(ckpt['model_state'])
            model.eval()
            MODELS['nilm'][app_name] = {
                'model': model,
                'window': ckpt.get('window_size', 99),
                'n_states': ckpt['ns'],
                'scaler': scaler
            }
        except Exception as e:
            print(f"加载 NILM 模型 {app_name} 失败: {e}")
            return None
    return MODELS['nilm'].get(app_name)

def build_forecaster_features_20d(df_raw, max_power=6900.0):
    """
    完全对齐 forecaster_model.py 中 Energy_dataset (mode='refined') 的特征工程
    """
    df = df_raw.copy().sort_values('timestamp').reset_index(drop=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.resample('15min').mean().ffill().bfill()

    df['sin_hour'] = np.sin(2 * np.pi * df.index.hour / 24.0)
    df['cos_hour'] = np.cos(2 * np.pi * df.index.hour / 24.0)
    df['sin_month'] = np.sin(2 * np.pi * (df.index.month - 1) / 12.0)
    df['cos_month'] = np.cos(2 * np.pi * (df.index.month - 1) / 12.0)
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

    for lag in [1, 2, 3, 4]:
        df[f'lag_{lag}'] = df['whe_power'].shift(lag)
    df['diff_1'] = df['whe_power'].diff().fillna(0)

    for w in [2, 4, 12]:
        df[f'roll_max_{w}'] = df['whe_power'].rolling(window=w, min_periods=1).max()

    temp_col = 'Temperature' if 'Temperature' in df.columns else 'temperature'
    hum_col = 'Humidity' if 'Humidity' in df.columns else 'humidity'
    df['Temperature'] = df.get(temp_col, 20.0)
    df['Humidity'] = df.get(hum_col, 50.0)

    df = df.ffill().bfill().tail(96)

    whe_feat = (df[['whe_power']].values / max_power).astype(np.float32)


    base_cols = [
        'sin_hour', 'cos_hour', 'sin_month', 'cos_month', 'is_weekend',
        'lag_1', 'lag_2', 'lag_3', 'lag_4',
        'diff_1',
        'roll_max_2', 'roll_max_4', 'roll_max_12',
        'Temperature', 'Humidity'
    ]
    extra_feat = df[base_cols].values.astype(np.float32)

    for i, col in enumerate(base_cols):
        if 'lag' in col or 'diff' in col or 'roll' in col:
            extra_feat[:, i] /= max_power
        elif col == 'Temperature':
            extra_feat[:, i] = (extra_feat[:, i] + 20.0) / 60.0
        elif col == 'Humidity':
            extra_feat[:, i] /= 100.0

    nilm_feats = []
    for app in APPLIANCES_LIST:
        col = f'{app}_State'
        raw = df[col].values.astype(np.float32) if col in df.columns else np.zeros(len(df))
        max_s = max(float(raw.max()), 1.0)
        nilm_feats.append((raw / max_s).reshape(-1, 1))


    hpe_s = df['HPE_State'].values if 'HPE_State' in df.columns else np.zeros(len(df))
    dur_hpe = np.zeros(len(hpe_s))
    cnt1, prev1 = 0, -1
    for i, s in enumerate(hpe_s):
        if s > 0 and s == prev1:
            cnt1 += 1
        elif s > 0:
            cnt1 = 1
        else:
            cnt1 = 0
        dur_hpe[i] = cnt1
        prev1 = int(s)
    nilm_feats.append((dur_hpe / 96.0).clip(max=1.0).reshape(-1, 1))

    # CDE 持续时间
    cde_s = df['CDE_State'].values if 'CDE_State' in df.columns else np.zeros(len(df))
    dur_cde = np.zeros(len(cde_s))
    cnt2, prev2 = 0, -1
    for i, s in enumerate(cde_s):
        if s > 0 and s == prev2:
            cnt2 += 1
        elif s > 0:
            cnt2 = 1
        else:
            cnt2 = 0
        dur_cde[i] = cnt2
        prev2 = int(s)
    nilm_feats.append((dur_cde / 96.0).clip(max=1.0).reshape(-1, 1))

    all_feats = np.concatenate([whe_feat, extra_feat, *nilm_feats], axis=1)
    return torch.FloatTensor(all_feats).unsqueeze(0).to(DEVICE)

def build_nilm_features(df_raw, scaler, app_code):
    """
    构建与 NILM 训练完全一致的特征（使用 StandardScaler）
    """
    df = df_raw.copy().sort_values('timestamp').reset_index(drop=True)
    for col in df.columns:
        if col != 'timestamp':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.ffill().bfill()
    df['whe_power'] = df['whe_power'].clip(lower=0)

    idx = pd.to_datetime(df['timestamp'])
    hour = idx.dt.hour + idx.dt.minute / 60.0

    main_power_raw = df['whe_power'].values.reshape(-1, 1)
    diff_raw = np.diff(main_power_raw, axis=0, prepend=main_power_raw[0:1])

    temp = np.concatenate([main_power_raw, np.zeros_like(main_power_raw), diff_raw], axis=1)
    normalized = scaler.transform(temp)  
    main_p = normalized[:, 0:1]        
    diff_p = normalized[:, 2:3]         

    time_feats = [
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        (idx.dt.dayofweek >= 5).astype(np.float32)
    ]
    if app_code == 'HPE':
        month = idx.month
        time_feats.append(np.cos(2 * np.pi * (month - 1) / 12))

    time_feats = np.column_stack(time_feats).astype(np.float32)
    features = np.concatenate([main_p, diff_p, time_feats], axis=1)
    return torch.FloatTensor(features).unsqueeze(0).to(DEVICE)


def get_water_feature(ts, code, offset_hours=0, roll=False):
    target_ts = ts - timedelta(hours=offset_hours)
    target_str = target_ts.strftime('%Y-%m-%d %H:%M:%S')
    start_str = (target_ts - timedelta(hours=3)).strftime('%Y-%m-%d %H:%M:%S')
    table = 'water_main_logs' if code == "WHW" else 'water_sub_logs'
    fld = "AND app_code = :code" if code != "WHW" else ""
    params = {"target_str": target_str, "code": code}

    if roll:
        params["start_str"] = start_str
        q = f"""
                SELECT AVG(avg_rate) as v 
                FROM {table} 
                WHERE timestamp >= :start_str AND timestamp <= :target_str {fld}
            """
    else:
        q = f"""
                SELECT avg_rate as v 
                FROM {table} 
                WHERE timestamp <= :target_str {fld} 
                ORDER BY timestamp DESC LIMIT 1
            """
    res = pd.read_sql(text(q), engine, params=params)
    return float(res['v'].iloc[0]) if not res.empty and pd.notna(res['v'].iloc[0]) else 0.5


def _core_water_inference(dt):
    """内部函数：执行水务模型推理"""
    hour = dt.hour
    w_feat = [
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        dt.weekday(), (1 if dt.weekday() >= 5 else 0),
        (1 if 7 <= hour <= 9 else 0), (1 if 19 <= hour <= 22 else 0),
        (1 if 1 <= hour <= 5 else 0)
    ]
    for lag_h in [1, 24]:
        for code in ["WHW", "HTW", "DWW"]: w_feat.append(get_water_feature(dt, code, offset_hours=lag_h))
    for code in ["WHW", "HTW", "DWW"]:
        w_feat.append(get_water_feature(dt, code, offset_hours=1, roll=True))
    prediction = MODELS['water_gas']['water'].predict([w_feat])[0]
    return float(prediction[0])


def _core_gas_inference(avg_temp):
    """内部函数：执行燃气模型推理"""
    gas_val = MODELS['water_gas']['gas'].predict([[avg_temp]])[0]
    return float(gas_val)


def _get_daily_avg_temperature(current_datetime: str) -> float:
    """获取指定日期当天的平均温度"""
    dt = pd.to_datetime(current_datetime)
    date_s = dt.strftime('%Y-%m-%d')
    q_temp = """
                 SELECT AVG(temperature) as avg_temp 
                 FROM telemetry_main 
                 WHERE timestamp >= :start_time AND timestamp <= :end_time
             """
    df_temp = pd.read_sql(text(q_temp), engine, params={"start_time": f"{date_s} 00:00:00", "end_time": current_datetime})
    avg_t = df_temp['avg_temp'].iloc[0]

    if pd.notna(avg_t):
        return round(float(avg_t), 1)
    else:
        q_recent = """
                       SELECT AVG(temperature) as avg_temp 
                       FROM (
                            SELECT temperature 
                            FROM telemetry_main 
                            WHERE timestamp <= :end_time
                            ORDER BY timestamp DESC 
                            LIMIT 96
                       )
                   """
        try:
            recent_t = pd.read_sql(text(q_recent), engine, params={"end_time": current_datetime})['avg_temp'].iloc[0]
            return round(float(recent_t), 1) if pd.notna(recent_t) else 15.0
        except Exception:
            return 15.0


def _run_forecaster_inference(target_datetime: str):
    """提取历史数据并运行电力预测模型，返回 P10, P50, P90 数组"""
    if MODELS['forecaster'] is None:
        initialize_all_resources()

    q_hist = """
                 SELECT * FROM telemetry_main 
                 WHERE timestamp <= :target_datetime 
                 ORDER BY timestamp DESC LIMIT 1500
             """
    df_h = pd.read_sql(text(q_hist), engine, params={"target_datetime": target_datetime})
    if len(df_h) < 96:
        return None, None, None

    max_p = MODELS['forecaster_meta'].get('max_power', 6900.0)
    x = build_forecaster_features_20d(df_h, max_power=max_p)
    with torch.no_grad():
        out = MODELS['forecaster'](x)[0].cpu().numpy()

    p10_series = out[:, 0] * max_p
    p50_series = out[:, 1] * max_p
    p90_series = out[:, 2] * max_p
    return p10_series, p50_series, p90_series

@mcp.tool()
def realtime_nilm_monitor(user_app_name: str, target_datetime: str):
    """获取电器实时功率与额定参数。诊断运行效率。"""
    try:
        app_code = APPLIANCE_MAP.get(user_app_name, user_app_name)
        q_meta = """
                     SELECT nominal_power_w 
                     FROM appliances 
                     WHERE app_code = :app_code
                 """
        df_meta = pd.read_sql(text(q_meta), engine, params={"app_code": app_code})

        if df_meta.empty:
            return f"诊断失败：系统的电力网络中并未接入或录入“{user_app_name}”这个设备，无法读取其实时功率。请提示用户这可能是一个未联网的独立小家电。"

        nominal_w = df_meta.iloc[0, 0]

        model_info = get_nilm_model(app_code)
        if not model_info:
            return f"无法深入分析“{user_app_name}”的用电指纹，底层缺少针对该设备的 NILM 监测模型。"

        q_win = """
                    SELECT timestamp, whe_power 
                    FROM telemetry_main 
                    WHERE timestamp <= :target_datetime 
                    ORDER BY timestamp DESC LIMIT 99
                """
        df_win = pd.read_sql(text(q_win), engine, params={"target_datetime": target_datetime})
        if len(df_win) < 99:
            return f"数据不足，目前只有 {len(df_win)} 条记录，诊断需要 99 条。"

        latest_log_time = pd.to_datetime(df_win.iloc[0]['timestamp'])
        dt = pd.to_datetime(target_datetime)
        if (dt - latest_log_time).total_seconds() / 60.0 > 60:
            return f"【状态警告】当前处于日前预测模式，或设备已断线超过 1 小时（最新实测数据停留在 {latest_log_time.strftime('%m-%d %H:%M')}）。\n无法进行实时 NILM 功率分解与健康诊断，请基于预测走势为用户解答。"

        scaler = model_info.get('scaler')
        if scaler is not None:
            x_tensor = build_nilm_features(df_win, scaler, app_code)
        else:
            return f"NILM 模型缺少归一化参数，请重新训练 {user_app_name}。"

        with torch.no_grad():
            p_pred, s_logits = model_info['model'](x_tensor)
            state_idx = torch.argmax(s_logits[0], dim=1).item()
            actual_power = max(0.0, float(np.mean(p_pred.cpu().numpy())) * 6900.0)

        return (f"诊断数据 {user_app_name}：\n"
                f"设备额定功率：{nominal_w}W\n"
                f"实时推断功率：{actual_power:.1f}W\n"
                f"NILM 状态索引：{state_idx}\n"
                f"请作为诊断师：判断当前功率相对于额定值是否正常，并解释目前的运行强度。")
    except Exception as e:
        return f"诊断工具故障: {str(e)}"

@mcp.tool()
def get_current_weather_and_temperature(current_datetime: str):
    avg_temp = _get_daily_avg_temperature(current_datetime)
    return f"当前平均温度：{avg_temp}℃。"

@mcp.tool()
@lru_cache(maxsize=32)
def query_historical_energy(start_date: str, end_date: str = None):
    """精确查询过去一段时间的全屋用电量及耗电明细。如果是查询某一天，start_date 和 end_date 请传入相同的日期（如 '2012-05-15'）。"""
    if end_date is None or end_date == "":
        end_date = start_date

    try:
        q_main = """
                     SELECT SUM(whe_power)/60.0/1000.0 as total_sum 
                     FROM telemetry_main 
                     WHERE timestamp >= :start_time AND timestamp <= :end_time
                 """
        total_kwh = pd.read_sql(text(q_main), engine, params={"start_time": f"{start_date} 00:00:00", "end_time": f"{end_date} 23:59:59"})['total_sum'].iloc[0]
        total_kwh = total_kwh if pd.notna(total_kwh) else 0.0

        q_app = """
                    SELECT app_code, SUM(inferred_power)/60.0/1000.0 as kwh 
                    FROM nilm_inferred_logs 
                    WHERE timestamp >= :start_time AND timestamp <= :end_time 
                    GROUP BY app_code
                """
        df_app = pd.read_sql(text(q_app), engine, params={"start_time": f"{start_date} 00:00:00", "end_time": f"{end_date} 23:59:59"})

        app_details = [f"{REVERSE_APPLIANCE_MAP.get(r['app_code'], r['app_code'])}: {r['kwh']:.2f} 度" for _, r in
                       df_app.iterrows()]
        app_str = "，".join(app_details) if app_details else "无特定电器明细"

        return f"历史查询结果：{start_date} 至 {end_date}\n全屋总用电：{total_kwh:.2f} 度\n核心电器：{app_str}\n请准确汇报。"
    except Exception as e:
        return f"查询失败: {str(e)}"
def write_progress(filename, current, total, status, message=""):
    try:
        with open("pdf_progress.json", "w", encoding="utf-8") as f:
            json.dump({
                "filename": filename,
                "current": current,
                "total": total,
                "status": status,
                "message": message
            }, f)
    except:
        pass
def _background_learn_pdf(app_name: str, file_path: str):
    try:
        logger.info(f"正在阅读: {file_path}")

        if MODELS['vectorstore'] is None or MODELS['ocr'] is None:
            initialize_all_resources(mode="tools")

        if MODELS['vectorstore'] is None or MODELS['ocr'] is None:
            write_progress(app_name, 0, 0, "error", message="模型初始化失败，请检查显存。")
            return

        filename = os.path.basename(file_path)
        doc = fitz.open(file_path)
        all_docs = []
        total_pages = len(doc)

        write_progress(app_name, 0, total_pages, "started")

        for page_num in range(total_pages):
            write_progress(app_name, page_num + 1, total_pages, "processing")
            logger.info(f"正在解析【{app_name}】: 第 {page_num + 1}/{total_pages} 页...")
            page = doc[page_num]
            mat = fitz.Matrix(200 / 72, 200 / 72)
            pix = page.get_pixmap(matrix=mat)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)

            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            result, _ = MODELS['ocr'](img)

            if result:
                text_lines = [line[1] for line in result]
                raw_text = "\n".join(text_lines)

                if len(raw_text.strip()) >= 50:
                    cleaned = raw_text
                    if MODELS['translator_chain'] is not None:
                        try:
                            response = MODELS['translator_chain'].invoke({"text": raw_text})
                            cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
                            if "</OUTPUT>" in response:
                                cleaned = response.split("</OUTPUT>")[0].strip()
                            else:
                                cleaned = response.strip()
                                cleaned = re.sub(r'^(好的|明白|这就|以下是).*?[:：\n]', '', cleaned, flags=re.IGNORECASE)
                            if len(cleaned) < 15:
                                cleaned = raw_text
                        except Exception as translate_e:
                            logger.info(f"第 {page_num + 1} 页翻译超时或失败，保留原文。")

                    app_code = APPLIANCE_MAP.get(app_name, "UNKNOWN")
                    doc_obj = Document(
                        page_content=cleaned,
                        metadata={
                            "source": filename,
                            "page": page_num + 1,
                            "appliance_code": app_code,
                            "appliance_name": app_name
                        }
                    )
                    all_docs.append(doc_obj)

        doc.close()

        if all_docs:
            MODELS['vectorstore'].add_documents(all_docs)
            logger.info(f"【{app_name}】的说明书（共 {len(all_docs)} 页有效内容）已永久存入知识库！")
        write_progress(app_name, total_pages, total_pages, "completed")

    except Exception as e:

        error_msg = str(e)

        write_progress(app_name, 0, 0, "error", message=error_msg)

        logger.error(f"后台任务崩溃详情如下：\n{traceback.format_exc()}")

@mcp.tool()
def add_pdf_manual(app_name: str, file_path: str):
    """读取本地的 PDF 说明书文件，进行 OCR 识别、翻译并录入到知识库中。参数 app_name: 家电名称（如“智能烤箱”）。参数 file_path: PDF 文件的完整路径。"""
    if not os.path.exists(file_path):
        return f"录入失败：系统找不到文件，请检查路径是否正确 ({file_path})。"
    write_progress(app_name, 0, 100, "started", message="正在初始化解析引擎...")
    thread = threading.Thread(target=_background_learn_pdf, args=(app_name, file_path))
    thread.start()

    return f"指令已成功下达！我已经把【{app_name}】的说明书放到后台排队了。因为需要逐页进行视觉识别和翻译，大概需要几分钟时间。在此期间，您可以随时问我其他问题，互不影响！"

@mcp.tool()
def predict_today_consumption(current_datetime: str):
    """获取全屋今日已用电量、剩余时间预测及历史同期日均值。对比数据，分析今日用电是否存在异常并给出诊断建议。"""
    try:
        if len(current_datetime.strip()) <= 10:
            current_datetime = f"{current_datetime.strip()} 14:30:00"
        dt = pd.to_datetime(current_datetime)
        date_s = dt.strftime('%Y-%m-%d')

        q_real = """
                             SELECT timestamp, whe_power 
                             FROM telemetry_main 
                             WHERE timestamp >= :start_time AND timestamp <= :end_time
                         """
        df_real = pd.read_sql(text(q_real), engine, params={"start_time": f"{date_s} 00:00:00", "end_time": current_datetime}).sort_values('timestamp')

        last_idx = -1
        if not df_real.empty:
            df_real['ts'] = pd.to_datetime(df_real['timestamp']).dt.tz_localize(None)
            df_real['whe_power'] = pd.to_numeric(df_real['whe_power'], errors='coerce')

            latest_ts = df_real['ts'].max()
            last_idx = (latest_ts.hour * 60 + latest_ts.minute) // 15

            real_today_kwh = (
                    df_real.set_index('ts')['whe_power'].resample('15min').mean().ffill().sum() * 0.25 / 1000.0
            )
        else:
            real_today_kwh = 0.0

        if MODELS['forecaster'] is None: initialize_all_resources()

        q_hist = """
                         SELECT * FROM telemetry_main 
                         WHERE timestamp <= :current_datetime 
                         ORDER BY timestamp DESC LIMIT 1500
                     """
        x = build_forecaster_features_20d(pd.read_sql(text(q_hist), engine, params={"current_datetime": current_datetime}))

        with torch.no_grad():
            p50_output = MODELS['forecaster'](x)[0, :, 1].cpu().numpy() * 6900.0

        start = last_idx + 1 if last_idx != -1 else 0
        intervals_left = max(0, 96 - start)
        future_kwh = np.sum(p50_output[:intervals_left]) * 0.25 / 1000.0

        q_avg = """
                    SELECT date(timestamp) as day, SUM(whe_power)/60.0/1000.0 as daily_sum 
                    FROM telemetry_main 
                    WHERE timestamp < :start_time AND timestamp >= :past_time 
                    GROUP BY date(timestamp)
                """
        df_avg = pd.read_sql(text(q_avg), engine, params={"start_time": f"{date_s} 00:00:00", "past_time": str(dt - timedelta(days=30))})
        hist_avg_kwh = df_avg['daily_sum'].mean() if not df_avg.empty else 12.5

        if last_idx == -1:
            today_kwh_str = "暂无实测数据 (日前预测模式)"
            prompt_instruction = (
                f"【系统强制警告：今天尚未发生任何实际用电记录！】\n"
                f"严禁在回复中计算时间进度！严禁说'用电少'或'进度慢'！请明确告知用户当前为预测模式，并直接利用【历史日均({hist_avg_kwh:.1f}度)】给出排班建议。"
            )
        else:
            today_kwh_str = f"{real_today_kwh:.2f} 度"
            time_pct = int(start / 96 * 100)
            prompt_instruction = (
                f"请作为【能源分析师】：对比今日已用({real_today_kwh:.1f}度)与历史日均({hist_avg_kwh:.1f}度)。\n"
                f"结合目前数据进度(今日已跑完 {time_pct}%)，分析用电进度是否过快，并给出全天是否会超支的诊断。"
            )

        return (f"全屋能耗分析报告 ({dt.strftime('%m月%d日 %H:%M')})：\n"
                f"今日已消耗：{today_kwh_str}\n"
                f"预计今日剩余消耗：{future_kwh:.2f} 度\n"
                f"历史全天平均消耗：{hist_avg_kwh:.2f} 度\n"
                f"{prompt_instruction}")
    except Exception as e:
        return f"能耗分析工具调用失败: {str(e)}"

@mcp.tool()
def predict_next_24_hours(current_datetime: str):
    """获取未来24小时的预测曲线特征。根据峰值时间建议用户避峰用电。"""
    try:
        if len(current_datetime.strip()) <= 10:
            current_datetime = f"{current_datetime.strip()} 14:30:00"
        dt = pd.to_datetime(current_datetime)

        p10_series, p50_series, p90_series = _run_forecaster_inference(current_datetime)

        if p50_series is None:
            return "历史数据不足，无法进行未来 24 小时预测。"

        q_latest = """
                       SELECT timestamp 
                       FROM telemetry_main 
                       WHERE timestamp <= :current_datetime 
                       ORDER BY timestamp DESC LIMIT 1
                    """
        df_latest = pd.read_sql(text(q_latest), engine, params={"current_datetime": current_datetime})
        if not df_latest.empty:
            latest_ts = pd.to_datetime(df_latest['timestamp'].iloc[0])
            pred_start_ts = latest_ts + timedelta(minutes=15)
        else:
            pred_start_ts = dt

        avg_p50 = np.mean(p50_series)
        peak_indices = np.where(p50_series > avg_p50 * 1.5)[0]
        peaks = [(pred_start_ts + timedelta(minutes=int(i * 15))).strftime('%H:%M') for i in peak_indices[:5]]

        p10_kwh = np.sum(p10_series) * 0.25 / 1000.0
        p90_kwh = np.sum(p90_series) * 0.25 / 1000.0

        return (f"未来24小时预测特征：\n"
                f"总能耗区间：{p10_kwh:.2f} ~ {p90_kwh:.2f} kWh\n"
                f"预测负载均值：{avg_p50:.1f} W\n"
                f"疑似高峰时间点：{', '.join(peaks) if peaks else '无明显高峰'}\n"
                f"分析这些高峰点，结合当前时间，建议用户将哪些高能耗任务（如洗衣服、烘干）转移到低谷时段。")

    except Exception as e:
        return f"规划数据获取失败: {str(e)}"

@mcp.tool()
def predict_today_appliance_consumption(app_name: str, current_datetime: str):
    """获取特定电器的今日消耗、实时状态及历史均值。"""
    try:
        app_code = APPLIANCE_MAP.get(app_name, app_name)
        q_meta = """
                     SELECT app_code 
                     FROM appliances 
                     WHERE app_code = :app_code
                 """
        if pd.read_sql(text(q_meta), engine, params={"app_code": app_code}).empty:
            return f"分析失败：系统中并未录入“{app_name}”这个设备。"
        dt = pd.to_datetime(current_datetime)
        date_s = dt.strftime('%Y-%m-%d')

        q_today = """
                              SELECT SUM(inferred_power)/60.0/1000.0 as kwh 
                              FROM nilm_inferred_logs 
                              WHERE app_code=:app_code AND timestamp >= :start_time AND timestamp <= :end_time
                          """
        df_today = pd.read_sql(text(q_today), engine, params={"app_code": app_code, "start_time": f"{date_s} 00:00:00", "end_time": current_datetime})
        real_today_kwh = float(df_today['kwh'].iloc[0]) if not df_today.empty and pd.notna(
            df_today['kwh'].iloc[0]) else 0.0

        q_check = """
                      SELECT 1 
                      FROM telemetry_main 
                      WHERE timestamp >= :start_time AND timestamp <= :end_time LIMIT 1
                  """
        has_data_today = not pd.read_sql(text(q_check), engine, params={"start_time": f"{date_s} 00:00:00", "end_time": current_datetime}).empty

        model_info = get_nilm_model(app_code)
        state_idx = 0
        actual_power = 0.0

        if model_info and has_data_today:
            q_window = """
                           SELECT timestamp, whe_power 
                           FROM telemetry_main 
                           WHERE timestamp <= :end_time 
                           ORDER BY timestamp DESC LIMIT 99
                        """
            df_win = pd.read_sql(text(q_window), engine, params={"end_time": current_datetime})
            if len(df_win) >= 99:
                scaler = model_info.get('scaler')
                if scaler is None:
                    return f"NILM 模型 {app_name} 缺少归一化参数（scaler），请先运行 extract_nilm_scalers.py 提取。"
                x_tensor = build_nilm_features(df_win, scaler, app_code)
                with torch.no_grad():
                    p_pred, s_logits = model_info['model'](x_tensor)
                    state_idx = torch.argmax(s_logits[0], dim=1).item()
                    actual_power = max(0.0, float(np.mean(p_pred.cpu().numpy())) * 6900.0)

        q_hist = """
                     SELECT date(timestamp) as day, SUM(inferred_power)/60.0/1000.0 as daily_kwh 
                     FROM nilm_inferred_logs 
                     WHERE app_code=:app_code AND timestamp < :start_time AND timestamp >= :past_time 
                     GROUP BY date(timestamp)
                 """
        df_hist = pd.read_sql(text(q_hist), engine, params={"app_code": app_code, "start_time": f"{date_s} 00:00:00", "past_time": str(dt - timedelta(days=30))})
        hist_avg_kwh = df_hist['daily_kwh'].mean() if not df_hist.empty else 0.0

        if not has_data_today:
            status_desc = "预测模式 (无实测数据)"
            today_kwh_str = "暂无数据"
            actual_power_str = "未知"
            prompt_instruction = (
                f"【系统强制警告：当前为日前预测模式，该设备今天尚无真实耗电！】\n"
                f"严禁说该设备今天'节约'或'未使用'。请基于【历史每日平均】和【未来预测走势】给出建议。"
            )
        else:
            status_desc = "正在运行" if state_idx > 0 else "处于待机或关闭状态"
            today_kwh_str = f"{real_today_kwh:.3f} 度"
            actual_power_str = f"{actual_power:.1f}W"
            prompt_instruction = "请结合当前时间分析是否超标，并基于目前状态预测今日表现。"

        return (f"电器【{app_name}】数据分析报告：\n"
                f"今日累计已用：{today_kwh_str}\n"
                f"即时运行状态：{status_desc} (状态索引: {state_idx}, 即时功率: {actual_power_str})\n"
                f"历史每日平均：{hist_avg_kwh:.3f} 度\n"
                f"{prompt_instruction}")
    except Exception as e:
        return f"数据分析工具调用失败: {str(e)}"


@mcp.tool()
def predict_water_usage(current_datetime: str):
    """预测全屋实时用水速率、单日耗水预估及月度耗水预估。"""
    if MODELS['water_gas']['water'] is None:
        return "水预测模型未加载，请检查 checkpoints/water_multi_model.pkl 是否存在。"
    try:
        if MODELS['water_gas']['water'] is None: initialize_all_resources()
        dt = pd.to_datetime(current_datetime)
        date_s = dt.strftime('%Y-%m-%d')
        hour = dt.hour

        q_real_today = """
                           SELECT SUM(avg_rate) as used_so_far 
                           FROM water_main_logs 
                           WHERE timestamp >= :start_time AND timestamp <= :end_time
                       """
        try:
            df_real = pd.read_sql(text(q_real_today), engine, params={"start_time": f"{date_s} 00:00:00", "end_time": current_datetime})
            used_so_far = float(df_real['used_so_far'].iloc[0]) if not df_real.empty and pd.notna(df_real['used_so_far'].iloc[0]) else 0.0
        except Exception:
            used_so_far = 0.0

        q_check = """
                      SELECT 1 
                      FROM water_main_logs 
                      WHERE timestamp >= :start_time LIMIT 1
                   """
        try:
            has_data_today = not pd.read_sql(text(q_check), engine, params={"start_time": f"{date_s} 00:00:00"}).empty
        except Exception:
            has_data_today = False

        current_rate_lh = _core_water_inference(dt)

        if not has_data_today:
            q_avg_hist = """
                             SELECT SUM(avg_rate) as daily 
                             FROM water_main_logs 
                             GROUP BY date(timestamp)
                         """
            try:
                df_avg = pd.read_sql(text(q_avg_hist), engine)
                daily_est_l = df_avg['daily'].mean() if not df_avg.empty else 150.0
            except Exception:
                daily_est_l = 150.0
            used_so_far_str = "暂无实测数据 (日前预测模式)"
        else:
            intervals_left = 24 - hour - 1
            if intervals_left > 0:
                q_future_avg = """
                                   SELECT AVG(hourly_sum) as future_avg 
                                   FROM (SELECT date(timestamp) as d, SUM(avg_rate) as hourly_sum 
                                         FROM water_main_logs 
                                         WHERE strftime('%H', timestamp) > :hour_str AND timestamp >= :past_time 
                                   GROUP BY date(timestamp))
                               """
                try:
                    df_future = pd.read_sql(text(q_future_avg), engine, params={"hour_str": f"{hour:02d}", "past_time": str(dt - timedelta(days=30))})
                    future_est = float(df_future['future_avg'].iloc[0]) if not df_future.empty and pd.notna(df_future['future_avg'].iloc[0]) else (current_rate_lh * 0.3 * intervals_left)
                except Exception:
                    future_est = current_rate_lh * 0.3 * intervals_left
            else:
                future_est = 0.0
            daily_est_l = used_so_far + current_rate_lh + future_est
            used_so_far_str = f"{used_so_far:.1f} 升"

        monthly_est_ton = (daily_est_l * 30) / 1000.0

        return (
            f"实时水务分析报告（结合AI预测与水表实体数据）：\n"
            f"全屋当前瞬时流速：{current_rate_lh:.1f} 升/小时\n"
            f"今日0点至今已确切消耗：{used_so_far_str}\n"
            f"科学耗水预估：结合您的作息规律与历史数据推算，预计全天总耗水约为 {daily_est_l:.1f} 升，整月(30天)预估耗水约 {monthly_est_ton:.2f} 吨(立方米)。"
        )
    except Exception as e:
        return f"用水量预测失败: {str(e)}"

@mcp.tool()
def predict_gas_usage(current_datetime: str):
    """预测本月燃气总需求量。当用户询问燃气量、用气情况时调用。"""
    try:
        if MODELS['water_gas']['gas'] is None: initialize_all_resources()
        dt = pd.to_datetime(current_datetime)
        date_s = dt.strftime('%Y-%m-%d')

        q_temp = "SELECT AVG(temperature) as avg_temp FROM telemetry_main WHERE timestamp >= :start_time AND timestamp <= :end_time"
        df_temp = pd.read_sql(text(q_temp), engine, params={"start_time": f"{date_s} 00:00:00", "end_time": current_datetime})
        avg_temp = df_temp['avg_temp'].iloc[0]
        current_temp = round(float(avg_temp), 1) if pd.notna(avg_temp) else 15.0
        gas_val = _core_gas_inference(current_temp)

        q_avg_gas = "SELECT AVG(consumption_gj) FROM natural_gas_logs"
        try:
            avg_gas = pd.read_sql(text(q_avg_gas), engine).iloc[0, 0]
            avg_gas = float(avg_gas) if pd.notna(avg_gas) else 0.0
        except Exception:
            avg_gas = 0.0

        return (
            f"燃气系统分析：\n"
            f"当前环境温度：{current_temp} ℃\n"
            f"受气温影响，预计本月燃气总需求：{gas_val:.2f} GJ (作为参考，历史月均需求为: {avg_gas:.2f} GJ)\n"
        )
    except Exception as e:
        return f"燃气预测失败: {str(e)}"

@mcp.tool()
def predict_future_bills(resource_type: str, current_datetime: str, selected_date: str, days: int = 30, daily_energy_kwh: float = 0.0):
    """预估未来账单（包含电费、燃气费、水费）。"""
    year = pd.to_datetime(current_datetime).year
    try:
        calc_type_msg = ""
        dt = pd.to_datetime(current_datetime)
        date_s = dt.strftime('%Y-%m-%d')

        r_type = resource_type.lower()
        is_gas = r_type in ["gas", "燃气", "天然气", "气费"]
        is_water = r_type in ["water", "水", "水费", "自来水"]

        if daily_energy_kwh <= 0.0:
            if is_gas:
                avg_temp = _get_daily_avg_temperature(current_datetime)
                monthly_gas_gj = _core_gas_inference(avg_temp)
                daily_energy_val = monthly_gas_gj / 30.0
                calc_type_msg = f"按今日气温({avg_temp}℃)推算的日均燃气需求 {daily_energy_val:.3f} GJ计算"

            elif is_water:
                rate_lh = _core_water_inference(dt)
                daily_energy_val = (rate_lh * 24) / 1000.0
                calc_type_msg = f"按此时段AI模型推算的日均耗水 {daily_energy_val:.2f} 吨计算"

            else:
                if days == 1:
                    q_real = """
                                 SELECT SUM(whe_power)/60.0/1000.0 as kwh 
                                 FROM telemetry_main 
                                 WHERE timestamp >= :start_time AND timestamp <= :end_time
                             """
                    df_real = pd.read_sql(text(q_real), engine, params={"start_time": f"{date_s} 00:00:00", "end_time": current_datetime})
                    real_today_kwh = float(df_real['kwh'].iloc[0]) if not df_real.empty and pd.notna(
                        df_real['kwh'].iloc[0]) else 0.0

                    q_latest = """
                                   SELECT timestamp 
                                   FROM telemetry_main 
                                   WHERE timestamp >= :start_time AND timestamp <= :end_time 
                                   ORDER BY timestamp DESC LIMIT 1
                               """
                    df_latest = pd.read_sql(text(q_latest), engine, params={"start_time": f"{date_s} 00:00:00", "end_time": current_datetime})
                    if not df_latest.empty:
                        latest_ts = pd.to_datetime(df_latest['timestamp'].iloc[0])
                        last_idx = (latest_ts.hour * 60 + latest_ts.minute) // 15
                    else:
                        last_idx = -1

                    _, p50_series, _ = _run_forecaster_inference(current_datetime)
                    if p50_series is not None:
                        start = last_idx + 1 if last_idx != -1 else 0
                        future_kwh = np.sum(p50_series[:max(0, 96 - start)]) * 0.25 / 1000.0
                    else:
                        future_kwh = 0.0

                    daily_energy_val = real_today_kwh + future_kwh
                    calc_type_msg = "按今日真实已用+预测剩余耗电计算" if last_idx != -1 else "无实测数据，按全天24小时AI预测耗电计算"
                else:
                    q_avg = """
                                SELECT date(timestamp) as day, SUM(whe_power)/60.0/1000.0 as daily_sum 
                                FROM telemetry_main 
                                WHERE timestamp < :start_time AND timestamp >= :past_time 
                                GROUP BY date(timestamp)
                            """
                    df_avg = pd.read_sql(text(q_avg), engine, params={"start_time": f"{date_s} 00:00:00", "past_time": str(dt - timedelta(days=30))})
                    df_avg['daily_sum'] = pd.to_numeric(df_avg['daily_sum'], errors='coerce')
                    daily_energy_val = df_avg['daily_sum'].mean() if not df_avg.empty else 12.5
                    calc_type_msg = f"按历史日均耗电 {daily_energy_val:.2f} 度计算"
        else:
            daily_energy_val = daily_energy_kwh
            calc_type_msg = f"按传入的指定日均用量 {daily_energy_val:.2f} 计算"


        total_usage = daily_energy_val * days

        if is_gas:
            daily_base_fee = 29.58 / 30.0
            cost = (daily_base_fee * days) + (total_usage * 1.3385)
            return f"账单预估：预计 {days} 天的燃气费用约为 {cost:.2f} 加币 ({calc_type_msg})。"

        elif is_water:
            year = dt.year
            if year == 2012:
                daily_rate = 2.72
            elif year == 2013:
                daily_rate = 3.10
            elif year >= 2014:
                daily_rate = 3.26
            else:
                daily_rate = 0.0
            cost = daily_rate * days
            return f"账单预估：预计 {days} 天的水费约为 {cost:.2f} 加币 (基于 {year} 年固定费率包干计算)。"

        else:
            daily_v = total_usage / days if days > 0 else 0.0
            cost = 0.0
            step1_daily_threshold = 22.2

            for i in range(days):
                eval_date = dt + timedelta(days=i)

                if eval_date < pd.to_datetime('2012-04-01'):
                    basic, step1, step2, rider = 0.14480, 0.06670, 0.09620, 0.025
                    transit, tax_rate, credit_rate = 0.06240, 0.12, 0.07
                elif eval_date < pd.to_datetime('2013-04-01'):
                    basic, step1, step2, rider = 0.15050, 0.06800, 0.10190, 0.050
                    transit, tax_rate, credit_rate = 0.06240, 0.12, 0.07
                elif eval_date < pd.to_datetime('2014-04-01'):
                    basic, step1, step2, rider = 0.15270, 0.06900, 0.10340, 0.050
                    transit, tax_rate, credit_rate = 0.06240, 0.05, 0.00
                else:
                    basic, step1, step2, rider = 0.16640, 0.07520, 0.11270, 0.050
                    transit, tax_rate, credit_rate = 0.06240, 0.05, 0.00

                step1_kwh = min(step1_daily_threshold, daily_v)
                step2_kwh = max(0, daily_v - step1_daily_threshold)

                energy_charge = basic + (step1_kwh * step1) + (step2_kwh * step2)
                energy_with_rider = energy_charge * (1 + rider)

                daily_subtotal = energy_with_rider + transit
                daily_tax = daily_subtotal * tax_rate
                daily_credit = energy_with_rider * credit_rate

                cost += (daily_subtotal + daily_tax - daily_credit)

            return f"账单预估：结合加拿大BC省历史阶梯电价与精准税制，预计 {days} 天的电费为 {cost:.2f} 加币 ({calc_type_msg})。"

    except Exception as e:
        return f"账单预估失败: {str(e)}"


@mcp.tool()
def query_appliance_manual(app_name: str, problem_desc: str):
    """查阅家电说明书。"""
    try:
        if MODELS['vectorstore'] is None:
            initialize_all_resources(mode="tools")
        
        vectorstore = MODELS['vectorstore']
        bm25 = MODELS['bm25_index']
        bm25_docs = MODELS['bm25_docs']
        reranker = MODELS['reranker']
        search_filter = {"appliance_name": app_name} if app_name else None
        
        vector_results = vectorstore.similarity_search_with_score(
            problem_desc, 
            k=15,
            filter=search_filter
        )
        candidates = [doc for doc, _ in vector_results]

        if bm25 is not None and bm25_docs is not None:
            tokenized_query = problem_desc.split()
            bm25_scores = bm25.get_scores(tokenized_query)
            top_indices = np.argsort(bm25_scores)[-10:][::-1]

            existing_texts = {doc.page_content for doc in candidates}
            for idx in top_indices:
                text = bm25_docs[idx]
                if text not in existing_texts:

                    temp_doc = Document(page_content=text, metadata={"source": "bm25_recall"})
                    candidates.append(temp_doc)
                    existing_texts.add(text)
        
        if not candidates:
            return f"未在说明书中找到相关内容。"
        candidates = candidates[:10]
        final_docs = candidates
        if reranker is not None:
            try:
                pairs = [[problem_desc, doc.page_content] for doc in candidates]
                scores = reranker.compute_score(pairs, normalize=True)
                sorted_idx = np.argsort(scores)[::-1][:3]  # 取前 3 名
                final_docs = [candidates[i] for i in sorted_idx]
                logger.info(f"Rerank 完成，从 {len(candidates)} 个候选中精选出 {len(final_docs)} 个")
            except Exception as e:
                logger.warning(f"Rerank 执行失败，降级使用前 3 个候选: {e}")
                final_docs = candidates[:3]
        else:
            final_docs = candidates[:3]

        cleaned = []
        for d in final_docs:
            text = d.page_content.replace("#", "").replace("*", "").strip()
            source = d.metadata.get('appliance_name', '未知设备')
            cleaned.append(f"关于{source}的内容提示：{text[:300]}...")
        
        if not cleaned:
            return "未找到相关知识点。"
        
        return "已找到相关知识点：\n" + "\n".join(cleaned)
        
    except Exception as e:
        traceback.print_exc()
        return f"检索失败: {str(e)}"


@mcp.tool()
def delete_appliance_knowledge(app_name: str):
    """从知识库中彻底删除特定电器的说明书内容。"""
    try:
        if MODELS['vectorstore'] is None:
            initialize_all_resources(mode="tools")

        db = MODELS['vectorstore']
        collection = db._collection
        all_data = collection.get()
        ids_to_delete = []

        for i, metadata in enumerate(all_data['metadatas']):
            if (metadata.get('appliance_name') == app_name or
                    app_name.lower() in metadata.get('source', '').lower()):
                ids_to_delete.append(all_data['ids'][i])

        if ids_to_delete:
            db.delete(ids=ids_to_delete)
            logger.info(f"知识库清理成功：匹配到 {len(ids_to_delete)} 条记录")
            return f"已成功从我的大脑中清理了关于【{app_name}】的全部 {len(ids_to_delete)} 条知识点。"

        db.delete(where={"appliance_name": app_name})
        return f"清理指令已下达：虽然没找到匹配的 ID，但已尝试执行针对【{app_name}】的清理操作。"

    except Exception as e:
        return f"清理过程发生异常: {str(e)}"

@mcp.tool()
def get_home_energy_snapshot(current_time: str):
    """一键获取全屋能耗全景快照。"""
    try:
        dt = pd.to_datetime(current_time)
        date_s = dt.strftime('%Y-%m-%d')

        q_temp = """
                     SELECT AVG(temperature) as avg_temp 
                     FROM telemetry_main 
                     WHERE timestamp >= :start_time AND timestamp <= :end_time
                 """
        df_temp = pd.read_sql(text(q_temp), engine, params={"start_time": f"{date_s} 00:00:00", "end_time": current_time})
        avg_temp = df_temp['avg_temp'].iloc[0]
        actual_temp = round(float(avg_temp), 1) if pd.notna(avg_temp) else 15.0

        electricity_data = predict_today_consumption(current_datetime=current_time)

        apps_to_check = ["热泵", "烤箱", "干衣机"]
        app_reports = [predict_today_appliance_consumption(app_name=app, current_datetime=current_time) for app in apps_to_check]

        water_report = predict_water_usage(current_datetime=current_time)
        gas_report = predict_gas_usage(current_datetime=current_time)

        return (
            f"【全屋电力数据概览】\n{electricity_data}\n\n"
            f"【核心电器明细】\n{' | '.join(app_reports)}\n\n"
            f"【环境与生活水气资源】（当前动态参考温度：{actual_temp}℃）\n"
            f"{water_report}\n\n"
            f"{gas_report}\n\n"
            f"任务：为用户撰写一份专业且具有行动建议的家庭能源简报。"
        )
    except Exception as e:
        return f"快照获取失败: {str(e)}"

@mcp.tool()
async def global_inference_engine(target_datetime: str):
    """内部API专用：提供确定性的高频数据输出"""
    if MODELS['forecaster'] is None or MODELS['water_gas']['water'] is None:
        initialize_all_resources(mode="all")

    dt = pd.to_datetime(target_datetime)
    results = {"timestamp": target_datetime, "system_status": "normal", "alerts": []}

    q_hist = """
                 SELECT * FROM telemetry_main 
                 WHERE timestamp <= :end_time 
                 ORDER BY timestamp DESC LIMIT 96
             """
    df_h = pd.read_sql(text(q_hist), engine, params={"end_time": target_datetime})

    if not df_h.empty:
        max_p = MODELS['forecaster_meta'].get('max_power', 6900.0)
        x_fore = build_forecaster_features_20d(df_h, max_power=max_p)
        with torch.no_grad():
            pred_out = MODELS['forecaster'](x_fore)[0].cpu().numpy()
            p50_preds = pred_out[:, 1] * max_p
            p90_preds = pred_out[:, 2] * max_p
            results["elec_forecast_kw"] = [round(float(x / 1000.0), 2) for x in p50_preds[:24]]

            actual_power = df_h.iloc[0]['whe_power']
            last_log_time = pd.to_datetime(df_h.iloc[0]['timestamp'])
            predicted_p90 = float(p90_preds[0])
            time_diff_minutes = (dt - last_log_time).total_seconds() / 60.0

            if time_diff_minutes <= 60 and actual_power > predicted_p90 * 1.2:
                results["system_status"] = "warning"
                results["alerts"].append({
                    "type": "anomaly_detection",
                    "msg": f"异常用电：当前功率({actual_power}W)远超模型预测上限({predicted_p90:.2f}W)"
                })

            future_peak = np.max(p50_preds[:24])
            if future_peak > 4500:
                results["alerts"].append({
                    "type": "control_recommendation",
                    "action": "load_shifting",
                    "msg": "预测到即将出现用电高峰，建议向 Agent 发出削峰信号（如暂停干衣机）"
                })

    water_rate = _core_water_inference(dt)
    q_temp = """
            SELECT AVG(temperature) as t 
            FROM (
                SELECT temperature FROM telemetry_main 
                WHERE timestamp <= :end_time 
                ORDER BY timestamp DESC LIMIT 5
            )
        """
    avg_t = pd.read_sql(text(q_temp), engine, params={"end_time": target_datetime})['t'].iloc[0]
    avg_t = float(avg_t) if pd.notna(avg_t) else 15.0
    gas_gj = _core_gas_inference(avg_t)

    results["water_inference"] = {"current_rate_lh": round(water_rate, 2)}
    results["gas_inference"] = {"monthly_est_gj": round(gas_gj, 2), "ref_temp_c": round(avg_t, 1)}

    return json.dumps(results, ensure_ascii=False)


@mcp.tool()
async def get_dashboard_chart_data(selected_time: str):
    """内部API专用：为前端提供 96 维绘图数组、运行时间轴及KPI指标"""
    print(f"DEBUG: 接收到的时间是 {selected_time}")

    if MODELS['forecaster'] is None or MODELS['water_gas']['water'] is None:
        initialize_all_resources(mode="dashboard")

    try:
        dt = pd.to_datetime(selected_time)
        selected_date = dt.strftime('%Y-%m-%d')

        target_apps = ['HPE', 'BME', 'CDE', 'CWE', 'WOE', 'FGE', 'TVE']
        app_timeline = {app: [0] * 96 for app in target_apps}
        forecast_data = {"p10": [None] * 96, "p50": [None] * 96, "p90": [None] * 96}
        real_data = [None] * 96

        query_real = """
                         SELECT timestamp, whe_power 
                         FROM telemetry_main 
                         WHERE timestamp >= :start_time AND timestamp <= :end_time 
                         ORDER BY timestamp ASC
                     """
        df_real = pd.read_sql(text(query_real), engine, params={"start_time": f"{selected_date} 00:00:00", "end_time": selected_time})
        last_idx = -1

        if not df_real.empty:
            df_real['ts'] = pd.to_datetime(df_real['timestamp'])
            for _, row in df_real.iterrows():
                idx = row['ts'].hour * 4 + row['ts'].minute // 15
                if 0 <= idx < 96:
                    val = row['whe_power']
                    real_data[idx] = round(float(val), 1) if pd.notna(val) else 0.0
                    last_idx = idx

        p10_series, p50_series, p90_series = _run_forecaster_inference(selected_time)

        if p50_series is not None:
            start = last_idx + 1 if last_idx != -1 else 0
            for j in range(start, 96):
                safe_idx = min(j - start, len(p50_series) - 1)
                forecast_data['p10'][j] = round(float(p10_series[safe_idx]), 1)
                forecast_data['p50'][j] = round(float(p50_series[safe_idx]), 1)
                forecast_data['p90'][j] = round(float(p90_series[safe_idx]), 1)
            if last_idx != -1:
                forecast_data['p10'][last_idx] = real_data[last_idx]
                forecast_data['p50'][last_idx] = real_data[last_idx]
                forecast_data['p90'][last_idx] = real_data[last_idx]

                appliance_thresholds = {
                    'CDE': {'k': 100, 'h': 300},
                    'WOE': {'k': 100, 'h': 600, 'true_off': 150},
                    'HPE': {'k': 25, 'h': 60, 'true_off': 30},
                    'CWE': {'k': 30, 'h': 200, 'true_off': 20},
                    'FGE': {'k': 50, 'h': 100, 'true_off': 40},
                    'BME': {'k': 30, 'h': 60},
                    'TVE': {'k': 50, 'h': 120, 'true_off': 60}
                }

                query_logs = """
                    SELECT app_code, timestamp, inferred_power 
                    FROM nilm_inferred_logs 
                    WHERE timestamp >= :start_time AND timestamp <= :end_time
                """
                df_logs = pd.read_sql(text(query_logs), engine, params={"start_time": f"{selected_date} 00:00:00", "end_time": selected_time})
                if not df_logs.empty:
                    df_logs['ts'] = pd.to_datetime(df_logs['timestamp'])
                    for _, row in df_logs.iterrows():
                        code = row['app_code']
                        if code in app_timeline:
                            t_idx = row['ts'].hour * 4 + row['ts'].minute // 15

                            if 0 <= t_idx < 96:
                                config = appliance_thresholds.get(code, {})
                                threshold = config.get('true_off', config.get('k', 20))

                                if row['inferred_power'] > threshold:
                                    app_timeline[code][t_idx] = 1

        query_nilm = """
                         SELECT app_code, inferred_power 
                         FROM nilm_inferred_logs 
                         WHERE timestamp >= :start_time AND timestamp <= :end_time 
                         ORDER BY timestamp DESC 
                         LIMIT 50
                     """
        df_nilm = pd.read_sql(text(query_nilm), engine, params={"start_time": f"{selected_date} 00:00:00", "end_time": selected_time})
        nilm_res = []

        if not df_nilm.empty:
            df_latest = df_nilm.drop_duplicates(subset=['app_code'], keep='first')

            for _, row in df_latest.iterrows():
                app_name = REVERSE_APPLIANCE_MAP.get(row['app_code'], row['app_code'])
                power_w = row['inferred_power']

                if pd.notna(power_w) and power_w > 5:
                    nilm_res.append({
                        "name": app_name,
                        "value": round(float(power_w), 1)
                    })

        valid_real = [v for v in real_data if v is not None]

        if last_idx == -1 and p50_series is not None:
            total_kwh = np.sum(p50_series) * 0.25 / 1000.0
        else:
            total_kwh = sum(valid_real) * 0.25 / 1000.0 if valid_real else 0.0

        kpi_metrics = {
            "today_kwh": round(total_kwh, 2),
            "estimated_cost": round(total_kwh * 0.12, 2),
            "carbon_kg": round(total_kwh * 0.42, 2)
        }

        water_rate = _core_water_inference(dt)
        q_water_today = """
                    SELECT SUM(avg_rate) as used 
                    FROM water_main_logs 
                    WHERE timestamp >= :start_time AND timestamp <= :end_time
                """
        try:
            water_used = pd.read_sql(text(q_water_today), engine, params={"start_time": f"{selected_date} 00:00:00", "end_time": selected_time})['used'].iloc[0]
            water_used = float(water_used) if pd.notna(water_used) else 0.0
        except Exception:
            water_used = 0.0


        avg_t = _get_daily_avg_temperature(selected_time)
        gas_pred = _core_gas_inference(avg_t)

        is_prediction_mode = (last_idx == -1)
        results = {
            "forecast": forecast_data,
            "real_data": real_data,
            "nilm": nilm_res if nilm_res else [{"name": "暂无数据", "value": 0}],
            "timeline": app_timeline,
            "kpis": kpi_metrics,
            "water_gas": {
                "water_current_rate": round(water_rate, 2),
                "water_today_total": round(float(water_used), 1),
                "gas_monthly_pred": round(gas_pred, 2),
                "current_temp": avg_t
            },
            "display_mode": "PREDICTION" if is_prediction_mode else "REALTIME",
            "system_warning": "WARNING_PREDICTION_MODE" if is_prediction_mode else None
        }

        return json.dumps(results, ensure_ascii=False)

    except Exception as e:
        traceback.print_exc()
        return json.dumps({"error": str(e)}, ensure_ascii=False)
    
def reciprocal_rank_fusion(bm25_results, vector_results, k=60):
    """
    倒数排名融合（RRF）
    bm25_results: list of (doc_index, score) 
    vector_results: list of (doc_id, doc_text, metadata, score) 
    """
    scores = {}
    
    for rank, (idx, _) in enumerate(bm25_results, start=1):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank)

    for rank, (doc_id, _, _, _) in enumerate(vector_results, start=1):
        pass 
    return scores

def hybrid_search(query, top_k=10):
    vectorstore = MODELS['vectorstore']
    bm25 = MODELS['bm25_index']
    bm25_docs = MODELS['bm25_docs']
    
    if vectorstore is None:
        return []

    vector_results = vectorstore.similarity_search_with_score(query, k=10)

    if bm25 is None or bm25_docs is None:
        return [doc for doc, _ in vector_results][:top_k]
    
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_indices = np.argsort(bm25_scores)[-10:][::-1]
    top_bm25_scores = [bm25_scores[i] for i in top_bm25_indices]

    doc_map = {}  
    for doc, score in vector_results:
        doc_map[doc.page_content] = {"doc": doc, "v_score": score, "bm25_score": -1}
    
    for idx, bm25_score in zip(top_bm25_indices, top_bm25_scores):
        text = bm25_docs[idx]
        if text in doc_map:
            doc_map[text]["bm25_score"] = bm25_score
        else:
            pass 

    merged_docs = []
    seen_texts = set()
    
    for doc, _ in vector_results:
        if doc.page_content not in seen_texts:
            merged_docs.append(doc)
            seen_texts.add(doc.page_content)

    for idx in top_bm25_indices:
        text = bm25_docs[idx]
        if text not in seen_texts:
 
            from langchain_core.documents import Document
            temp_doc = Document(page_content=text, metadata={})
            merged_docs.append(temp_doc)
            seen_texts.add(text)
    
    return merged_docs[:10]  

if __name__ == "__main__":
    mcp.run()