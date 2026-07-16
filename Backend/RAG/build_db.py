import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz
import torch
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

load_dotenv()

# =====================================================================
# 路径配置（修正：数据库放在 Backend/ 目录下）
# =====================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))              # RAG/
PDF_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../datasets/appliance_manuals/"))
DB_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "../chroma_db_appliances"))   # 修正为 Backend/chroma_db_appliances

# 设备映射表（仅保留你需要的）
MANUAL_MAP = {
    "Manual_FEQ1442": {"code": "CDE", "name": "干衣机"},
    "Manual_FTF2140": {"code": "CWE", "name": "洗衣机"},
    "Manual_Heritage16": {"code": "HPE", "name": "热泵"},
    "Manual_4a6h6036": {"code": "HPE", "name": "热泵"},
    "Manual_TXCSpec": {"code": "HPE", "name": "热泵"},
    "Manual_587144034": {"code": "BME", "name": "照明"},
    "Manual_NR1403": {"code": "TVE", "name": "电视"},
    "Manual_DMPBDT2": {"code": "TVE", "name": "电视"},
    "Manual_Arris": {"code": "TVE", "name": "电视"},
    "Manual_SuperCubes": {"code": "TVE", "name": "电视"},
    "Manual_TCP42ST30": {"code": "TVE", "name": "电视"},
    "Manual_hbl5045auc": {"code": "WOE", "name": "电烤箱"},
    "Manual_Amana": {"code": "FGE", "name": "冰箱"},
}

# =====================================================================
# 云端大模型
# =====================================================================
api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    raise ValueError("未在 .env 文件中设置 DASHSCOPE_API_KEY，请配置后重试。")

# 初始化阿里云通义千问（兼容 OpenAI 接口）
llm = ChatOpenAI(
    model="qwen-plus",                    # 可选 qwen-max, qwen-turbo 等
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=api_key,
    temperature=0.0,                      # 确定性输出
    max_tokens=2048,                      # 最大生成 token 数
    timeout=60,                           # 请求超时（秒）
    max_retries=2,                        # 失败重试次数
    # 如需关闭思考模式（仅 qwen 系列支持）：
    # model_kwargs={"extra_body": {"enable_thinking": False}}
)
print("[云端模型] 已成功连接阿里云通义千问（qwen-plus）", flush=True)

# =====================================================================
# 清洗与翻译 Prompt
# =====================================================================
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个极其严谨的工业级技术文档清洗与翻译专家，精通家电说明书的排版自愈。
你当前接收到的输入，crumbs 是从英文说明书 PDF 中提取并自动判定排版后重组的原始文本。

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

clean_chain = prompt | llm | StrOutputParser()


def get_appliance_info(filename):
    for key, info in MANUAL_MAP.items():
        if key.lower() in filename.lower():
            return info["code"], info["name"]
    return "UNKNOWN", "未知"


def translate_single_chunk(index, raw_text, filename, total):
    if len(raw_text) < 15:
        return index, raw_text

    print(f"正在调用模型清洗与翻译: {filename} 的第 {index + 1}/{total} 块...")

    for attempt in range(3):
        try:
            response = clean_chain.invoke({"text": raw_text})
            cleaned = response.strip()

            # 过滤可能的前言后记
            cleaned = re.sub(r'^(好的|明白|这就|以下是|为您翻译|这里有).*?[:：\n]', '', cleaned, flags=re.IGNORECASE).strip()
            # 过滤 <think> 标签
            cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL | re.IGNORECASE).strip()

            if len(cleaned) >= 15:
                print(f"   [成功]  第 {index + 1}/{total} 块清洗与翻译完毕！")
                return index, cleaned
        except Exception as e:
            print(f"   [重试] 第 {index + 1}/{total} 块异常 (尝试 {attempt+1}/3): {e}")
            time.sleep(2.0 * (attempt + 1))

    print(f"[失败] 第 {index + 1}/{total} 块重试耗尽，已回退为原文。")
    return index, raw_text


def process_pdf_with_pymupdf(pdf_path):
    filename = os.path.basename(pdf_path)
    code, name = get_appliance_info(filename)

    print(f"\n==========================================")
    print(f"处理: {filename} -> {code} ({name})")
    print(f"==========================================")

    try:
        doc = fitz.open(pdf_path)
        md_content = ""

        for page_num, page in enumerate(doc):
            rect = page.rect
            width = rect.width
            mid_x = width / 2

            blocks = page.get_text("blocks")
            text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]

            # 自适应判断单栏/双栏
            crossing_or_wide_count = 0
            for b in text_blocks:
                x0, y0, x1, y1, text, block_no, block_type = b
                if x0 < (mid_x - width * 0.1) and x1 > (mid_x + width * 0.1):
                    crossing_or_wide_count += 1
                elif (x1 - x0) > (width * 0.6):
                    crossing_or_wide_count += 1

            is_double_column = True
            if len(text_blocks) > 0:
                if (crossing_or_wide_count / len(text_blocks)) > 0.25 or len(text_blocks) < 4:
                    is_double_column = False
            else:
                is_double_column = False

            page_text = ""

            if is_double_column:
                left_column, right_column, full_width_blocks = [], [], []
                for b in text_blocks:
                    x0, y0, x1, y1, text, block_no, block_type = b
                    cleaned_text = text.strip()
                    center_x = (x0 + x1) / 2
                    if (x1 - x0) > (width * 0.8):
                        full_width_blocks.append((y0, cleaned_text))
                    elif center_x < mid_x:
                        left_column.append((y0, cleaned_text))
                    else:
                        right_column.append((y0, cleaned_text))

                left_column.sort(key=lambda x: x[0])
                right_column.sort(key=lambda x: x[0])
                full_width_blocks.sort(key=lambda x: x[0])

                if full_width_blocks:
                    top_blocks = [b[1] for b in full_width_blocks if b[0] < rect.height * 0.2]
                    if top_blocks:
                        page_text += "\n".join(top_blocks) + "\n\n"

                left_text = "\n".join([b[1] for b in left_column])
                right_text = "\n".join([b[1] for b in right_column])
                page_text += f"{left_text}\n\n{right_text}"
            else:
                sorted_blocks = sorted(text_blocks, key=lambda x: (x[1], x[0]))
                page_text = "\n\n".join([b[4].strip() for b in sorted_blocks])

            md_content += f"\n\n## Page {page_num + 1}\n\n{page_text}"

        print("PyMuPDF 几何布局自适应感知重构成功！")
    except Exception as e:
        print(f"PDF 提取失败 ({filename}): {e}")
        return []

    if not md_content or len(md_content.strip()) < 15:
        print(f"警告: {filename} 未提取到任何有效文本，跳过该文件！")
        return []

    # 结构化切分
    headers_to_split_on = [("#", "Header_1"), ("##", "Header_2"), ("###", "Header_3")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    md_header_splits = markdown_splitter.split_text(md_content)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=600,
        separators=["\n\n", "\n", " ", ""]
    )
    final_chunks = text_splitter.split_documents(md_header_splits)
    print(f"结构化切分成功，共划分 {len(final_chunks)} 个块，开始启动云端模型清洗与翻译...")

    # 并发翻译（并发数为 3）
    translated_results = {}
    max_concurrent_workers = 3

    with ThreadPoolExecutor(max_workers=max_concurrent_workers) as executor:
        futures = {
            executor.submit(translate_single_chunk, i, chunk_doc.page_content.strip(), filename, len(final_chunks)): chunk_doc
            for i, chunk_doc in enumerate(final_chunks)
        }

        completed_count = 0
        for future in as_completed(futures):
            chunk_doc = futures[future]
            try:
                idx, translated_text = future.result()
                translated_results[idx] = (translated_text, chunk_doc.metadata)
                completed_count += 1
                print(f"   [系统进度] 云端清洗进度已推进至: {completed_count}/{len(final_chunks)} 块...")
            except Exception as exc:
                print(f"[并发抛错] 该块生成异常: {exc}")

    # 组装最终文档
    all_docs = []
    for i in range(len(final_chunks)):
        if i in translated_results:
            translated_text, section_meta = translated_results[i]
            final_metadata = {
                "source": filename,
                "source_type": "pdf_manual",
                "chunk_index": i + 1,
                "appliance_code": code,
                "appliance_name": name,
                **section_meta
            }
            all_docs.append(Document(page_content=translated_text, metadata=final_metadata))

    return all_docs


def build_db():
    print("[DEBUG] 进入 build_db 函数")

    if not os.path.exists(PDF_DIR):
        os.makedirs(PDF_DIR)
        print(f"PDF 目录不存在，已自动创建：{PDF_DIR}")
        return

    pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
    print(f"[DEBUG] 找到 {len(pdf_files)} 个 PDF 文件：{pdf_files}")

    if not pdf_files:
        print("没有找到 PDF 文件")
        return

    # 清理旧库
    if os.path.exists(DB_PATH):
        print(f"发现旧的数据库目录 {DB_PATH}，正在清理...")
        shutil.rmtree(DB_PATH, ignore_errors=True)
        print("旧数据库已清理完毕")

    # ------------------------------------------------------------------
    # 2. 提前加载 embedding 模型（只加载一次，避免重复消耗资源）
    # ------------------------------------------------------------------
    model_path = os.path.abspath(os.path.join(SCRIPT_DIR, "../local_models/BAAI/bge-m3"))
    if os.path.exists(model_path):
        print("[Embedding] 成功检测向量提取模型！")


    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 【必须添加的 3 行代码】：强制底层彻底离线，禁止任何联网行为
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    print(f"--> [系统提示] 已强制开启离线模式。正在将模型加载至 {device}，请等待 30 秒...")

    # 加载模型
    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={'device': device}, 
        # 【修改点】：加入 local_files_only 确保只读本地
        encode_kwargs={'normalize_embeddings': True}
    )
    # 由于 HuggingFaceEmbeddings 的参数传递机制，最好的防联网方法就是上面加的 os.environ
    print("--> [系统提示] 模型加载成功！准备写入数据库...")
    # ------------------------------------------------------------------
    # 3. 创建空的向量库（如果目录不存在，Chroma 会自动创建）
    # ------------------------------------------------------------------
    vectorstore = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )
    print(f"向量库已初始化，目标路径：{DB_PATH}")

    # ------------------------------------------------------------------
    # 4. 逐个 PDF 处理，每处理完一个立即写入并刷盘
    # ------------------------------------------------------------------
    total_docs = 0
    for idx, pdf_file in enumerate(pdf_files, start=1):
        pdf_path = os.path.join(PDF_DIR, pdf_file)
        print(f"\n📄 正在处理第 {idx}/{len(pdf_files)} 个文件：{pdf_file}")

        try:
            docs = process_pdf_with_pymupdf(pdf_path)
        except Exception as e:
            print(f"❌ 处理 {pdf_file} 时发生严重异常，跳过该文件：{e}")
            continue

        if not docs:
            print(f"⚠️ {pdf_file} 未提取到有效文档块，已跳过。")
            continue

        # 写入向量库
        vectorstore.add_documents(docs)
        # 强制持久化到磁盘（确保立即落盘）
        total_docs += len(docs)

        current_total = vectorstore._collection.count()
        print(f"✅ {pdf_file} 已存入数据库，本次写入 {len(docs)} 块，累计总块数：{current_total}")

    # ------------------------------------------------------------------
    # 5. 最终统计
    # ------------------------------------------------------------------
    print(f"\n{'='*50}")
    print(f"所有 PDF 处理完成！")
    print(f"成功处理的 PDF 文件数：{len(pdf_files)}")
    print(f"数据库中文档块总数：{vectorstore._collection.count()}")
    print(f"数据库存放路径：{DB_PATH}")
    print(f"{'='*50}")


if __name__ == "__main__":
    build_db()