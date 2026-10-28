# 当前版本运行说明

## 环境与配置

建议使用 Python 3.12 和 Node.js 18+。在项目根目录创建独立环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r Backend/requirements.txt
python -m pip check
Copy-Item Backend/.env.example Backend/.env
```

编辑 `Backend/.env` 填写自己的密钥。主模型通过阿里云兼容接口调用，路由、专家和备用汇总使用 DeepSeek 官方接口；模型 ID、地域和访问权限必须与服务商账号匹配。前端切换的是主回复模型，不代表所有专家一起切换。不要提交 `.env`。

Windows 安装 `chroma-hnswlib` 时如果提示缺少 MSVC，需要安装 Microsoft C++ Build Tools 的“使用 C++ 的桌面开发”组件。不要盲目升级全部依赖；旧版 Pydantic、PyTorch 和新版 Transformers/PyArrow 的组合曾导致启动失败，当前 requirements 已明确约束这些版本。

## 本地资源

以下内容不随 Git 仓库提交，需要自行准备：

- `Backend/hems_data.db`：能源 SQLite 数据库。
- 整个 `Backend/chroma_db_appliances/`：已有说明书库与索引，不是只复制其中的 sqlite 文件。
- `Backend/local_models/BAAI/bge-base-zh-v1.5/`：本地嵌入模型。
- `Backend/checkpoints/`：负荷预测、各电器 NILM、水和气预测权重。
- 可选 `Backend/local_models/BAAI/bge-reranker-base/`，或 Hugging Face 的已有本地缓存。RAG 查询不会现场下载模型；无重排模型时回退到融合检索。可设置 `RAG_USE_RERANKER=false` 关闭重排，修改后重启后端。

根目录旧说明中的 `download_models.py` 仍是历史 Qwen 下载脚本，不适用于当前 BGE 检索资源，恢复已有系统时不要运行它。

已有数据库时不要运行 `setup_database.py`，该脚本会重建表。`RAG/build_db.py` 是历史批量建库脚本，仍有旧模型路径并会清理目标向量库，不能直接用于保留已有库的迁移。原始训练数据 `datasets/` 未提交，重新训练或重建数据需要单独准备。

## 启动

后端终端（项目根目录激活环境后）：

```powershell
cd Backend
python -X utf8 -m uvicorn app:app --host 127.0.0.1 --port 5000
```

另开前端终端：

```powershell
cd frontend
npm install
# 仅当 npm 提示这两个安装脚本尚未批准时执行：
npm approve-scripts esbuild vue-demi
npm run dev
```

前端访问 Vite 打印的 Local 地址；后端接口说明位于 `http://127.0.0.1:5000/docs`。两个终端保持开启。页面默认日期属于历史沙盒，应选择数据库中实际存在的日期；数据库记录及存储 NILM 快照不等同于实时电表读数。模型、提示和后端代码修改后需重启服务以清除智能体缓存。

## 本次修复

- 历史实测能源查询新增轻量 MCP 工具，不为简单查询执行全屋预测。
- 专家按需初始化；减少重复调度和摘要；增加阶段与工具耗时日志。
- 主模型超时不重复重试；最终汇总可切换 DeepSeek 备用服务。
- RAG 单独加载、本地缓存、设备过滤、向量与 BM25 融合；保留完整片段和引用。
- 普通说明书问答使用一次预设 MCP 检索与一次证据回答；上传/删除保留原管理流程。
- 回复语言跟随当前问题；建议与手册要求区分；前端区分超时、后端错误和断连。
- 为部分缺证据、异响和个人寿命/换新问题增加保守规则及汇总保护。这些是应用层规则，不是通用语义事实验证器。
- 预测输入特征调整到所提供 22 维权重的时间特征顺序，并保留维度检查。

## 验证边界

检索时会从历史文本中去除 `<think>` 翻译处理备注，但旧向量尚未重建。文件名和页码来自库内元数据，不代表已核验原 PDF 或印刷页码。预测特征中的状态聚合、归一化与历史训练流程仍需进一步核对，不能用图表能显示来证明预测准确性。上传建库、原文翻译、所有模型 ID 和所有外部 API 场景也不因这些代码修复而自动获得正确性保证。
