import json
import os
import pandas as pd
import re
import asyncio
import traceback
import shutil
import operator
import logging
from datetime import datetime
from typing import TypedDict, Annotated, Optional
from dotenv import load_dotenv
import sys
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["CHROMA_TELEMETRY_OTHERS"] = "False"
os.environ["USE_TORCH"] = "1"
os.environ["NO_TF"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
last_served_progress_status = "idle"
from fastapi import Form,FastAPI, HTTPException, Query, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

import yaml

import difflib
import uuid
from sqlalchemy import create_engine, inspect
from werkzeug.utils import secure_filename

VIRTUAL_DATE = os.getenv("VIRTUAL_SYSTEM_DATE", "2014-04-01")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "hems_data.db").replace("\\", "/")
DB_URI = f"sqlite:///{DB_PATH}"
db_engine = create_engine(DB_URI, connect_args={'check_same_thread': False})
def load_prompts():
    with open("config/prompts.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

PROMPTS = load_prompts()

global_mcp_tools = []
mcp_client = None
agent_cache = {}
global_mcp_session = None
mcp_lock = asyncio.Lock()
@asynccontextmanager
async def lifespan(app: FastAPI):
    global global_mcp_tools, mcp_client, global_mcp_session

    logger.info("\n正在连接MCP能源工具服务器...")
    mcp_client = MultiServerMCPClient({
        "energy_server": {
            "command": sys.executable,
            "args": ["-X", "utf8", "mcp_server.py"],
            "transport": "stdio"
        }
    })

    async with mcp_client.session("energy_server") as session:
        global_mcp_session = session
        global_mcp_tools = await load_mcp_tools(session)
        tools_names = [t.name for t in global_mcp_tools]
        logger.info(f"MCP工具拉取完毕，共有 {len(tools_names)} 个可用工具：\n{tools_names}")

        yield

    logger.info("服务正在关闭...")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"detail": "服务器内部错误，请稍后重试"}
    )

def truncate_history(history_str: str, max_len: int = 3000) -> str:
    if len(history_str) <= max_len:
        return history_str
    return history_str[:max_len] + "...(历史过长已截断)"

def clearable_reducer(left: list, right: list) -> list:
    if right and len(right) > 0 and right[0] == "CLEAR":
        return right[1:]
    return (left or []) + (right or [])

def latest_reducer(left: list, right: list) -> list:
    return right if right else (left or [])

class AgentState(TypedDict):
    question: str
    chat_history: Annotated[list[str], operator.add]
    conversation_summary: Annotated[list[str], latest_reducer]
    current_time: str
    next_agents: list[str]
    expert_results: Annotated[list[str], clearable_reducer]
    final_reply: str
    uploaded_file_path: Optional[str]
    system_has_alerts: bool

class InferenceRequest(BaseModel):
    selected_date: Optional[str] = None

class ChatRequest(BaseModel):
    message: str = ""
    selected_date: Optional[str] = None
    model: str = Field(default="qwen3:8b")
    session_id: str = "default"
    uploaded_file_path: Optional[str] = None
def get_or_create_multi_agent(model_name: str):
    if model_name in agent_cache:
        return agent_cache[model_name]

    logger.info(f"[{model_name}]正在查询可用子智能体...")

    if model_name == "qwen3.6-27b":
        main_llm = ChatOpenAI(
            model="qwen3.6-27b",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            temperature=0,
            timeout=15,
            model_kwargs={"extra_body": {"enable_thinking": False}}
        )
        router_llm = ChatOpenAI(
            model="deepseek-chat",
            base_url="https://api.deepseek.com",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            temperature=0,
            timeout=15,
        )
    elif model_name == "kimi-k2.6":
        main_llm = ChatOpenAI(
            model="kimi-k2.6",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            temperature=0,
            timeout=15,
            model_kwargs={"extra_body": {"enable_thinking": False}}
        )
        router_llm = ChatOpenAI(
            model="deepseek-chat",
            base_url="https://api.deepseek.com",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            temperature=0,
            timeout=15,
        )
    elif model_name == "qwen3.7-max-2026-05-20":
        main_llm = ChatOpenAI(
            model="qwen3.7-max-2026-05-20",
            base_url="https://api.deepseek.com",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            temperature=0,
            timeout=15,
        )
        router_llm = ChatOpenAI(
            model="deepseek-chat",
            base_url="https://api.deepseek.com",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            temperature=0,
            timeout=15,
        )
    else:
        main_llm = ChatOpenAI(
            model="qwen3.6-max-preview",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            temperature=0,
            timeout=15,
            model_kwargs={"extra_body": {"enable_thinking": False}}
        )
        router_llm = main_llm

    sub_llm = ChatOpenAI(
        model="deepseek-chat",
        base_url="https://api.deepseek.com",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        temperature=0,
        timeout=60,
    )

    def create_expert_executor(tools: list):
        system_message = "你是一个高效的AI智能助手，能够使用提供的工具回答用户问题。请准确地使用工具并返回有用的结果。"
        return create_react_agent(sub_llm, tools, prompt=system_message)
    
    async def summarize_conversation(llm, history: list[str], previous_summary: str = "") -> str:
        """生成或更新对话摘要 - 增量方式"""
        if len(history) < 3:
            return previous_summary or "对话刚刚开始。"
        
        recent = "\n".join(history[-7:]) 
        
        prompt = f"""你是一个家庭能源管家助手。请基于【历史摘要】和【新增对话】生成**更新后的完整摘要**。

                历史摘要：{previous_summary}

                新增对话：
                {recent}

                请突出以下内容：
                - 用户关注的核心设备/问题
                - 用户偏好或习惯
                - 已解决的事项和待办
                - 关键事实（日期、上传文件、特殊需求）

                输出格式：【摘要】... （严格控制在150字以内）"""
        
        try:
            resp = await llm.ainvoke([SystemMessage(content=prompt)])
            cleaned = resp.content.strip()
            if not cleaned.startswith("【摘要】"):
                cleaned = f"【摘要】{cleaned}"
            return cleaned
        except Exception as e:
            logger.warning(f"生成摘要失败: {e}")
            return previous_summary or "无摘要"

    def get_tool(name: str):
        for t in global_mcp_tools:
            if t.name == name: return t
        raise ValueError(f"MCP服务器未提供工具: {name}")

    analyst_exec = create_expert_executor([
        get_tool("get_home_energy_snapshot"),
        get_tool("predict_today_consumption"),
        get_tool("predict_today_appliance_consumption"),
        get_tool("query_historical_energy")
    ])

    planner_exec = create_expert_executor([
        get_tool("get_current_weather_and_temperature"),
        get_tool("predict_next_24_hours"),
        get_tool("predict_future_bills"),
        get_tool("predict_water_usage"),
        get_tool("predict_gas_usage")
    ])

    equipment_exec = create_expert_executor([
        get_tool("realtime_nilm_monitor")
    ])

    knowledge_exec = create_expert_executor([
        get_tool("query_appliance_manual"),
        get_tool("add_pdf_manual"),
        get_tool("delete_appliance_knowledge")
    ])

    async def supervisor_node(state: AgentState):
        logger.info("\n主管正在并发审视需求")
        
        history_list = state.get("expert_results", [])
        chat_history = state.get("chat_history", [])
        
        new_summary = ""
        
        summary_list = state.get("conversation_summary") or []
        prev_summary = summary_list[-1] if (isinstance(summary_list, list) and len(summary_list) > 0) else ""

        if len(chat_history) >= 4:
            new_summary = await summarize_conversation(router_llm, chat_history, prev_summary)
        else:
            new_summary = prev_summary

        if len(history_list) >= 5:
            logger.info("检测到多轮重复调用，触发断路器...")
            return {"next_agents": ["FINISH"], "conversation_summary": [new_summary]}

        history_str = "\n".join(history_list) if history_list else "无"
        history_str = truncate_history(history_str)

        recent_history = "\n".join(chat_history[-7:-1])
        summary_context = new_summary

        sys_prompt = PROMPTS["supervisor"].format(
            question=state['question'],
            history_str=history_str
        )
        
        full_prompt = f"""【对话摘要】{summary_context}

                          【近期对话】{recent_history}

                          {sys_prompt}""" if summary_context or recent_history else sys_prompt

        resp = await router_llm.ainvoke([SystemMessage(content=full_prompt)])
        content = resp.content.upper()
        valid_agents = ["ANALYST", "PLANNER", "EQUIPMENT", "KNOWLEDGE", "GENERAL", "FINISH"]
        raw_actions = [agent for agent in valid_agents if agent in content]

        worked_departments = []
        for expert_name, tag in [
            ("ANALYST", "[数据分析师报告]"),
            ("PLANNER", "[家庭规划师报告]"),
            ("EQUIPMENT", "[设备诊断专家报告]"),
            ("KNOWLEDGE", "[售后知识库报告]")
        ]:
            if tag in history_str:
                expert_block = history_str.split(tag)[-1]
                if not any(word in expert_block for word in ["无法", "失败", "未找到", "Error"]):
                    worked_departments.append(expert_name)

        next_actions = [act for act in raw_actions if act in valid_agents and act not in worked_departments]

        if state.get("system_has_alerts",
                     False) and "EQUIPMENT" not in next_actions and "EQUIPMENT" not in worked_departments:
            logger.info("检测到系统警报，强制拦截并唤醒设备专家 (EQUIPMENT)")
            next_actions.append("EQUIPMENT")
            if "GENERAL" in next_actions:
                next_actions.remove("GENERAL")
            if "FINISH" in next_actions:
                next_actions.remove("FINISH")

        if not next_actions or "FINISH" in next_actions:
            next_actions = ["FINISH"]

        logger.info(f"主管决定同时呼叫：{next_actions}")
        return {
                    "next_agents": next_actions,
                    "conversation_summary": [new_summary]   # 通过返回值更新
                }

    async def analyst_node(state: AgentState):
        logger.info("\n任务已派发给：数据分析师")
        history_list = state.get("expert_results", [])
        history_str = "\n".join(history_list) if history_list else "无"

        recent_chat = "\n".join(state.get("chat_history", [])[-5:])
        combined_history = f"【对话背景】\n{recent_chat}\n\n【本轮已获报告】\n{history_str}"

        sys_prompt = PROMPTS["analyst"].format(
            current_time=state['current_time'],
            question=state['question'],
            history=combined_history
        )
        res = await analyst_exec.ainvoke({"messages": [("user", sys_prompt)]})
        return {"expert_results": [f"[数据分析师报告]：\n{res['messages'][-1].content}"]}

    async def planner_node(state: AgentState):
        logger.info("\n任务已派发给：家庭规划师")
        history_list = state.get("expert_results", [])
        history_str = "\n".join(history_list) if history_list else "无"

        recent_chat = "\n".join(state.get("chat_history", [])[-5:])
        combined_history = f"【对话背景】\n{recent_chat}\n\n【本轮已获报告】\n{history_str}"

        sys_prompt = PROMPTS["planner"].format(
            current_time=state['current_time'],
            question=state['question'],
            history=combined_history
        )
        res = await planner_exec.ainvoke({"messages": [("user", sys_prompt)]})
        return {"expert_results": [f"[家庭规划师报告]：\n{res['messages'][-1].content}"]}

    async def equipment_node(state: AgentState):
        logger.info("\n任务已派发给：设备环境专家")
        history_list = state.get("expert_results", [])
        history_str = "\n".join(history_list) if history_list else "无"

        recent_chat = "\n".join(state.get("chat_history", [])[-5:])
        combined_context = f"【对话背景】\n{recent_chat}\n\n【本轮已获报告】\n{history_str}"

        sys_prompt = PROMPTS["equipment"].format(
            current_time=state['current_time'],
            question=state['question'],
            history=combined_context
        )
        res = await equipment_exec.ainvoke({"messages": [("user", sys_prompt)]})
        return {"expert_results": [f"[设备诊断专家报告]：\n{res['messages'][-1].content}"]}

    async def knowledge_node(state: AgentState):
        logger.info("\n任务已派发给：客服/知识库专员")
        history_list = state.get("expert_results", [])
        history_str = "\n".join(history_list) if history_list else "无"

        recent_chat = "\n".join(state.get("chat_history", [])[-5:])
        combined_context = f"【对话背景】\n{recent_chat}\n\n【本轮已获报告】\n{history_str}"

        sys_prompt = PROMPTS["knowledge"].format(
            question=state['question'],
            history=combined_context
        )
        final_input = sys_prompt
        if state.get("uploaded_file_path"):
            final_input += f"\n\n(系统提示：用户上传了文件，绝对路径为: {state['uploaded_file_path']})"
        res = await knowledge_exec.ainvoke({"messages": [("user", final_input)]})
        return {"expert_results": [f"[售后知识库报告]：\n{res['messages'][-1].content}"]}

    async def general_node(state: AgentState):
        logger.info("\n判定为日常闲聊")
        sys_prompt = PROMPTS["general"]
        res = await main_llm.ainvoke([SystemMessage(content=sys_prompt), HumanMessage(content=state["question"])])
        return {"expert_results": [f"[常规回复]：{res.content}"]}

    async def synthesizer_node(state: AgentState):
        """负责总结汇报 + 生成最终摘要"""
        all_reports = "\n\n".join(state.get("expert_results", []))
        if len(all_reports) > 6000:
            all_reports = all_reports[:6000] + "\n...(内容过长已截断)"

        chat_history = state.get("chat_history", [])
        recent_history = "\n".join(chat_history[-7:-1])
        
        summary_list = state.get("conversation_summary") or []
        prev_summary = summary_list[-1] if (isinstance(summary_list, list) and len(summary_list) > 0) else ""
        final_summary = await summarize_conversation(router_llm, chat_history, prev_summary)  # 使用 router_llm 降低成本

        sys_prompt = PROMPTS["synthesizer"].format(
            current_time=state['current_time'],
            all_reports=all_reports,
            question=state["question"]
        )

        full_prompt = f"""【对话摘要】{final_summary}

                          【近期对话】{recent_history}

                          {sys_prompt}"""

        try:
            res = await main_llm.ainvoke([SystemMessage(content=full_prompt), HumanMessage(content=state["question"])])
            clean_reply = res.content.replace('\u00a0', ' ')
            clean_reply = re.sub(r'(\n\|.*\|)\n(?![|])', r'\1\n\n', clean_reply)

            return {
                "final_reply": clean_reply,
                "chat_history": [f"管家：{clean_reply}"],
                "conversation_summary": [final_summary]
            }
        except Exception as e:
            logger.error(f"Synthesizer 错误: {e}")
            return {
                "final_reply": f"技术问题，原始数据：\n{all_reports}",
                "conversation_summary": [prev_summary]
            }

    workflow = StateGraph(AgentState)
    workflow.add_node("Supervisor", supervisor_node)
    workflow.add_node("Analyst", analyst_node)
    workflow.add_node("Planner", planner_node)
    workflow.add_node("Equipment", equipment_node)
    workflow.add_node("Knowledge", knowledge_node)
    workflow.add_node("General", general_node)
    workflow.add_node("Synthesizer", synthesizer_node)

    workflow.set_entry_point("Supervisor")
    workflow.add_conditional_edges(
        "Supervisor",
        lambda state: state["next_agents"],
        {"ANALYST": "Analyst", "PLANNER": "Planner", "EQUIPMENT": "Equipment", "KNOWLEDGE": "Knowledge",
         "GENERAL": "General", "FINISH": "Synthesizer"}
    )
    workflow.add_edge("General", "Synthesizer")
    for node in ["Analyst", "Planner", "Equipment", "Knowledge"]:
        workflow.add_edge(node, "Supervisor")
    workflow.add_edge("Synthesizer", END)

    memory = MemorySaver()
    compiled_agent = workflow.compile(checkpointer=memory)

    agent_cache[model_name] = compiled_agent
    return compiled_agent

UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    logger.info(f"\n[后端接收到文件]: {file.filename}")
    try:
        safe_name = secure_filename(file.filename or "uploaded_file")
        if not safe_name:
            safe_name = "uploaded_file"
        ext = os.path.splitext(safe_name)[1]
        unique_name = f"{uuid.uuid4().hex}{ext}"
        file_path = os.path.join(UPLOAD_DIR, unique_name)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        abs_path = os.path.abspath(file_path).replace("\\", "/")
        logger.info(f"[文件已保存至绝对路径]: {abs_path}")
        return {"file_path": abs_path, "message": "上传成功"}
    except Exception as e:
        logger.error(f"[上传严重错误]: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"文件上传失败: {str(e)}")

@app.get("/api/latest_date")
async def get_latest_date():
    """供前端大屏获取数据库中最新的遥测数据日期"""
    global VIRTUAL_DATE
    try:
        VIRTUAL_DATE = get_latest_virtual_date()
        return {"latest_date": VIRTUAL_DATE}
    except Exception as e:
        logger.info(f"获取最新日期失败: {e}")
        return {"latest_date": VIRTUAL_DATE}

def get_latest_virtual_date():
    """从数据库所有相关表中动态扫描绝对最新时间"""
    latest_stamps = []

    check_configs = [
        ("telemetry_main", "timestamp"),
        ("water_main_logs", "timestamp"),
        ("water_sub_logs", "timestamp"),
        ("natural_gas_logs", "month_date")
    ]

    for table, col in check_configs:
        try:
            query = f"SELECT {col} FROM {table} ORDER BY {col} DESC LIMIT 1"
            df = pd.read_sql(query, db_engine)
            if not df.empty and pd.notna(df[col].iloc[0]):
                latest_stamps.append(pd.to_datetime(df[col].iloc[0]))
        except Exception:
            pass

    if latest_stamps:
        absolute_latest = max(latest_stamps)
        return absolute_latest.strftime('%Y-%m-%d')

    return os.getenv("VIRTUAL_SYSTEM_DATE", "2014-04-01")


def fuzzy_match_column(df_columns, possible_names):
    """智能模糊匹配：支援精准匹配、包含匹配以及相似度纠错匹配"""
    df_cols_lower = [str(c).lower().strip() for c in df_columns]

    for possible_name in possible_names:
        name_lower = possible_name.lower().strip()

        if name_lower in df_cols_lower:
            idx = df_cols_lower.index(name_lower)
            return df_columns[idx]

        for i, col_lower in enumerate(df_cols_lower):
            if name_lower in col_lower or col_lower in name_lower:
                return df_columns[i]

        matches = difflib.get_close_matches(name_lower, df_cols_lower, n=1, cutoff=0.8)
        if matches:
            idx = df_cols_lower.index(matches[0])
            return df_columns[idx]

    return None

@app.post("/api/upload_dataset")
async def upload_dataset(
        dataset_type: str = Form(..., description="数据类型：electricity, water_main, water_sub, gas"),
        file: UploadFile = File(...)
):
    try:
        raw_name = file.filename or "dataset.csv"
        if not raw_name.endswith(".csv"):
            raise HTTPException(status_code=400, detail="目前只支持 CSV 文件")
        safe_name = secure_filename(raw_name)
        if not safe_name:
            safe_name = f"dataset_{uuid.uuid4().hex}.csv"
        file_path = os.path.join(UPLOAD_DIR, safe_name)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        df = pd.read_csv(file_path)

        table_name = ""
        rename_map = {}

        if dataset_type == "electricity":
            table_name = "telemetry_main"

            match_time = fuzzy_match_column(df.columns, ['time', 'timestamp', 'date/time', '时间', 'unix_ts'])
            match_power = fuzzy_match_column(df.columns, ['whe_true', 'whe_power', 'whe', '总功率', 'power'])
            match_temp = fuzzy_match_column(df.columns, ['temperature', 'temp (c)', 'temp', '温度', '气温'])
            match_hum = fuzzy_match_column(df.columns, ['humidity', 'hum', 'rel hum (%)', '湿度'])

            if match_time: rename_map[match_time] = 'timestamp'
            if match_power: rename_map[match_power] = 'whe_power'
            if match_temp: rename_map[match_temp] = 'temperature'
            if match_hum: rename_map[match_hum] = 'humidity'

            target_appliances = ['HPE', 'BME', 'CDE', 'CWE', 'WOE', 'TVE', 'FGE']
            for app in target_appliances:
                match_p_norm = fuzzy_match_column(df.columns, [f'{app}_Power_Norm', f'{app}_power', f'{app}功率'])
                match_state = fuzzy_match_column(df.columns, [f'{app}_State', f'{app}_status', f'{app}状态'])

                if match_p_norm: rename_map[match_p_norm] = f'{app}_Power_Norm'
                if match_state: rename_map[match_state] = f'{app}_State'

            df = df.rename(columns=rename_map)

            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
            else:
                raise ValueError("未找到时间列！")

            if 'temperature' not in df.columns: df['temperature'] = 20.0
            if 'humidity' not in df.columns: df['humidity'] = 50.0

        # 主水表
        elif dataset_type == "water_main":
            table_name = "water_main_logs"
            match_time = fuzzy_match_column(df.columns, ['unix_ts', 'timestamp', 'time', '时间'])
            match_counter = fuzzy_match_column(df.columns, ['counter', '累计读数', '总水量'])
            match_avg = fuzzy_match_column(df.columns, ['avg_rate', '平均流量', '流速'])
            match_inst = fuzzy_match_column(df.columns, ['inst_rate', '瞬时流量'])

            if match_time: rename_map[match_time] = 'timestamp'
            if match_counter: rename_map[match_counter] = 'counter'
            if match_avg: rename_map[match_avg] = 'avg_rate'
            if match_inst: rename_map[match_inst] = 'inst_rate'

            df = df.rename(columns=rename_map)
            if 'timestamp' in df.columns and pd.api.types.is_numeric_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

        # 子水表
        elif dataset_type == "water_sub":
            table_name = "water_sub_logs"
            match_time = fuzzy_match_column(df.columns, ['unix_ts', 'timestamp', 'time', '时间'])
            match_app = fuzzy_match_column(df.columns, ['app_code', '设备代号', '设备'])
            match_avg = fuzzy_match_column(df.columns, ['avg_rate', '平均流量'])

            if match_time: rename_map[match_time] = 'timestamp'
            if match_app: rename_map[match_app] = 'app_code'
            if match_avg: rename_map[match_avg] = 'avg_rate'

            df = df.rename(columns=rename_map)
            if 'timestamp' in df.columns and pd.api.types.is_numeric_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

            if 'app_code' not in df.columns:
                raise ValueError("子水表数据必须包含 app_code (设备代号) 列！")

        # 燃气表
        elif dataset_type == "gas":
            table_name = "natural_gas_logs"
            match_time = fuzzy_match_column(df.columns, ['month year', 'month_date', '月份', '日期'])
            match_gas = fuzzy_match_column(df.columns, ['net consumption (gj)', 'consumption_gj', '燃气消耗'])
            match_temp = fuzzy_match_column(df.columns, ['avg outside temp (℃)', 'avg_outside_temp', '平均气温'])

            if match_time: rename_map[match_time] = 'month_date'
            if match_gas: rename_map[match_gas] = 'consumption_gj'
            if match_temp: rename_map[match_temp] = 'avg_outside_temp'

            df = df.rename(columns=rename_map)
            if 'month_date' in df.columns:
                try:
                    df['month_date'] = pd.to_datetime(df['month_date'], format='%b %Y')
                except:
                    df['month_date'] = pd.to_datetime(df['month_date'])

        else:
            raise ValueError("未知的数据集类型")
        insp = inspect(db_engine)
        db_columns = [col['name'] for col in insp.get_columns(table_name)]

        if db_columns:
            cols_to_keep = [col for col in df.columns if col in db_columns]
            df_clean = df[cols_to_keep]
        else:
            df_clean = df

        df_clean.to_sql(name=table_name, con=db_engine, if_exists='append', index=False, chunksize=500)
        inserted_nilm_rows = 0

        if dataset_type == "electricity":
            nilm_records = []
            target_appliances = ['HPE', 'BME', 'CDE', 'CWE', 'WOE', 'TVE', 'FGE']

            for app in target_appliances:
                p_col = f'{app}_Power_Norm'
                s_col = f'{app}_State'

                if p_col in df_clean.columns and s_col in df_clean.columns:
                    app_df = df_clean[['timestamp', p_col, s_col]].copy()
                    app_df['app_code'] = app

                    app_df['inferred_power'] = app_df[p_col] * 6900.0
                    app_df['current_state'] = app_df[s_col].astype(int)

                    app_df = app_df[['timestamp', 'app_code', 'current_state', 'inferred_power']]
                    nilm_records.append(app_df)

            if nilm_records:
                final_nilm_df = pd.concat(nilm_records, ignore_index=True)
                final_nilm_df.to_sql(name='nilm_inferred_logs', con=db_engine, if_exists='append', index=False)
                inserted_nilm_rows = len(final_nilm_df)
                logger.info(f"联动写入 NILM 日志成功，共 {inserted_nilm_rows} 条记录")

        return {
            "message": "数据导入成功！",
            "inserted_rows": len(df_clean),
            "inserted_nilm_rows": inserted_nilm_rows,
            "table": table_name
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"导入失败: {str(e)}")


@app.post("/api/predict")
async def unified_predict(request: InferenceRequest):
    now_hms = datetime.now().strftime("%H:%M:%S")
    selected_time = request.selected_date or f"{VIRTUAL_DATE} {now_hms}"

    if not global_mcp_session:
        raise HTTPException(status_code=503, detail="MCP Service not ready")

    try:
        async with mcp_lock:
            result = await global_mcp_session.call_tool(
                "global_inference_engine",
                arguments={"target_datetime": selected_time}
            )

        if not result.content or len(result.content) == 0:
            raise ValueError("Empty response from MCP tool")

        data = json.loads(result.content[0].text)
        return {"code": 200, "message": "Inference completed", "data": data}
    except Exception as e:
        traceback.print_exc()

        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat(request: ChatRequest):
    user_message = request.message
    now_hms = datetime.now().strftime("%H:%M:%S")

    selected_time = request.selected_date or f"{VIRTUAL_DATE} {now_hms}"
    selected_model = request.model

    if len(selected_time.strip()) <= 10:
        selected_time = f"{selected_time.strip()} {now_hms}"
    try:
        system_has_alerts = False
        current_multi_agent = get_or_create_multi_agent(selected_model)
        inputs = {
            "question": user_message,
            "chat_history": [f"用户：{user_message}"],
            "current_time": selected_time,
            "uploaded_file_path": request.uploaded_file_path,
            "system_has_alerts": system_has_alerts,
            "expert_results": ["CLEAR"]
        }
        config = {
            "recursion_limit": 15,
            "configurable": {"thread_id": request.session_id}
        }
        result = await current_multi_agent.ainvoke(inputs, config=config)
        return {"reply": result["final_reply"]}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"暂时断线了：{str(e)}")


@app.get("/api/progress")
def get_pdf_progress():
    global last_served_progress_status
    try:
        progress_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdf_progress.json")
        if os.path.exists(progress_path):
            with open(progress_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            current_status = data.get("status", "idle")

            if current_status in ["completed", "error"]:
                if last_served_progress_status == current_status:
                    return {"status": "idle"}
                else:
                    last_served_progress_status = current_status
                    with open(progress_path, "w", encoding="utf-8") as f_reset:
                        json.dump({"status": "idle"}, f_reset)
                    return data

            last_served_progress_status = current_status
            return data

    except Exception as e:
        logger.info(f"读取进度文件异常: {e}")

    return {"status": "idle"}

@app.get("/api/dashboard")
async def dashboard(datetime_param: Optional[str] = Query(None, alias="datetime")):
    now_hms = datetime.now().strftime("%H:%M:%S")
    selected_time = datetime_param or f"{VIRTUAL_DATE} {now_hms}"
    if len(selected_time.strip()) <= 10:
        selected_time = f"{selected_time.strip()} {now_hms}"

    try:
        async with mcp_lock:
            result = await global_mcp_session.call_tool(
                "get_dashboard_chart_data",
                arguments={"selected_time": selected_time}
            )
        data = json.loads(result.content[0].text)
        return data
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Dashboard Error: {str(e)}")

if __name__ == '__main__':
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=False)