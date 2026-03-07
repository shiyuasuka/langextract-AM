# langextract-AM — 高熵合金论文结构化抽取 Pipeline

从高熵合金（HEA/MPEA）及增材制造（AM）领域学术 PDF 中抽取「成分–工艺–微观组织–力学性能」结构化数据，输出 JSONL，供下游机器学习或知识库使用。

采用 **PaddleOCR-VL 1.5 + LangExtract** 双引擎架构：先用 [PaddleOCR-VL 1.5](https://github.com/PaddlePaddle/PaddleOCR)（本地 GPU 部署）将 PDF 解析为高质量 Markdown 文本（表格自动转 Markdown），裁剪掉 Abstract / Introduction / References 等无关章节后保存为 `.txt`；再由 [LangExtract](https://github.com/google/langextract) 对清洗后的文本执行结构化抽取。默认按 **OpenAI 兼容接口** 调用 LLM，支持 `.env` 自定义 API Key / Base URL / Model。

---

## 功能概览

| 功能 | 说明 |
|------|------|
| **PaddleOCR-VL 预处理** | 本地部署 PaddleOCR-VL 1.5（SOTA 文档解析），PDF → Markdown（表格 / 公式 / 图表自动识别），裁剪 Abstract / Introduction / References，输出干净 `.txt` |
| **领域专家 Prompt** | 面向 HEA/MPEA + AM 的结构化抽取提示词：温度自动转 K、成分原子百分比拆解、工艺术语对齐（DED/LPBF/Arc Melting…）、各向异性方向分离 |
| **开放模型接入** | 支持任意 OpenAI 兼容 API（可直传 model_id）；Gemini 单独保留 |
| **分块抽取** | 按字符数分块 + 重叠，单块失败可切半重试；单块超时（默认 480s）跳过，不拖死整程 |
| **结构化输出** | 扁平 Extraction → 按 `material_id` 聚合为 MaterialEntity → 转目标 JSON 模板（`Composition_Info` / `Process_Info` / `Properties_Info`）写入 JSONL |
| **后处理标准化** | 工艺分类自动标准化（LENS/DED → AM_DED、SLM/LPBF → AM_LPBF、含热处理加 `_HeatTreated` 后缀）；化学式下标符号去除（Ti₄₂ → Ti42）；property_type 标准名映射；去重 |
| **去噪过滤** | `role ∈ {Target, Reference, Other}`，仅保留本文作者亲自制备和研究的 `Target` 材料，自动跳过 316L、Ti64 等对比 / 引用材料 |

---

## 项目结构

```
langextract-AM/
├── main.py                         # 入口：argparse、OCR 预处理、分块、lx.extract、聚合、写 JSONL
├── ocr_preprocess.py               # PaddleOCR-VL 1.5 预处理（PDF→Markdown→章节裁剪→.txt）
├── config_manager.py               # 模型工厂：环境变量驱动（OpenAI 兼容 + Gemini）
├── openai_compatible_provider.py   # 本地 OpenAI provider 扩展（支持 extra_body 等）
├── pdf_utils.py                    # PDF 提文本（PyMuPDF 备选）、clean_and_truncate_text、chunk_text
├── schemas.py                      # Pydantic 模型 + Prompt + 聚合 + 后处理 + 转目标 JSON
├── .env                            # 本地 API Key + 模型配置（不入库）
├── requirements.txt                # 依赖
├── AMpdf/                          # 待处理 PDF（及 OCR 生成的 .txt）
├── output/                         # 输出 he_data_{model}.jsonl
└── README.md                       # 本文件
```

### 处理流程

```
PDF ──► PaddleOCR-VL 1.5（本地 GPU）──► Markdown（表格/公式/图表自动识别）
    ──► 裁剪 Abstract / Introduction / References ──► .txt
    ──► 手动分块 ──► LangExtract 逐块抽取 ──► 聚合 MaterialEntity
    ──► 后处理（工艺标准化 / 化学式规格化 / 性能去重）──► JSONL
```

---

## 环境要求

| 项目 | 推荐配置 |
|------|----------|
| 操作系统 | Windows 10/11 |
| GPU | NVIDIA RTX 30/40 系列（PaddleOCR-VL 需要 CUDA） |
| Python | 3.10.x（推荐 3.10.10） |
| 包管理 | Anaconda / Miniconda |
| CUDA | 11.8 或 12.x（由 nvidia-smi 确认） |

> 没有 GPU 也可运行，见下文 [CPU 兜底方案](#9-cpu-兜底无-gpu-环境)。

---

## 安装（从零开始）

### 1. 前置准备

1. **安装 / 更新 NVIDIA 驱动**（通过 GeForce Experience 或 NVIDIA 官网），确认：

   ```bash
   nvidia-smi
   ```

   能看到 GPU 信息 + CUDA 版本（如 `CUDA Version: 11.8` 或 `12.x`）。

2. **安装 Git**：从 `https://git-scm.com/downloads` 下载，安装时勾选 "Add Git to PATH"。

3. **安装 Anaconda**：从 `https://www.anaconda.com/download` 下载安装。

### 2. 克隆仓库

```bash
cd D:\AI
git clone https://github.com/jiushiaaa/langextract-AM.git
cd langextract-AM
```

> 若仓库已存在，跳过克隆，每次使用前拉取最新代码即可：
>
> ```bash
> cd D:\AI\langextract-AM
> git pull origin main
> ```

### 3. 创建 conda 环境（Python 3.10）

```bash
conda create -n paddlepaddle python=3.10.10 -y
conda activate paddlepaddle
```

> 之后每次使用前先激活环境：`conda activate paddlepaddle`

### 4. 安装 PaddlePaddle-GPU

根据 `nvidia-smi` 中的 CUDA 版本选择：

```bash
# CUDA 11.8（推荐，兼容 RTX 30/40 系列）
python -m pip install "paddlepaddle-gpu==3.2.2" -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# CUDA 12.6
# python -m pip install "paddlepaddle-gpu==3.2.2" -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

自检：

```bash
python -c "import paddle; print('Paddle:', paddle.__version__); paddle.utils.run_check()"
```

看到 `PaddlePaddle is installed successfully!` 且提到 GPU 即表示 OK。

### 5. 安装 PaddleOCR + 项目依赖

```bash
# PaddleOCR（含 doc-parser）
python -m pip install "paddleocr[doc-parser]" -i https://pypi.tuna.tsinghua.edu.cn/simple

# 项目其它依赖
cd D:\AI\langextract-AM
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 6. 验证安装

```bash
python -c "import paddle; print('Paddle:', paddle.__version__); paddle.utils.run_check()"
python -c "import paddleocr; print('PaddleOCR:', paddleocr.__version__)"
python -c "import langextract, fitz, pdfplumber, dotenv; print('All deps OK')"
```

三条命令都不报错即通过。

---

## 配置

在项目根目录创建 `.env` 文件（不要提交到 Git）：

```env
# ═══ 必填：OpenAI 兼容接口 ═══
export LLM_API_KEY="bce-v3/xxx"
export LLM_BASE_URL="https://qianfan.bj.baidubce.com/v2"
export LLM_MODEL="ep_xxx_ernie-service"

# ═══ 推荐参数 ═══
export LLM_TEMPERATURE="0.1"
export LLM_MAX_OUTPUT_TOKENS="8192"
export LLM_ENABLE_THINKING="false"

# ═══ 可选参数 ═══
# PaddleOCR 引擎（默认 PaddleOCR-VL 1.5 GPU；无 GPU 设为 structurev3）
# export OCR_ENGINE="structurev3"

# 单块 LLM 超时（秒），默认 480；若经常出现超时跳过可适当加大
# export CHUNK_TIMEOUT_SECONDS="600"

# 思考预算（仅在 enable_thinking=true 时）
# export LLM_THINKING_BUDGET="800"

# 额外请求体（JSON 对象，可选）
# export LLM_EXTRA_BODY='{"enable_thinking": false}'

# 关闭强制 JSON 模式（部分网关不支持 response_format）
# export LLM_DISABLE_RESPONSE_FORMAT="true"

# 请求重试（默认重试 2 次，指数退避）
# export LLM_MAX_RETRIES="2"
# export LLM_RETRY_BACKOFF_SECONDS="1.5"

# 实体落地校验（默认 true，减少幻觉）
# export STRICT_ENTITY_GROUNDING="true"

# Gemini（仅在 --model gemini 时）
# export GOOGLE_API_KEY="你的Gemini密钥"
```

---

## 使用方法

### 快速开始（每次开机三步）

```bash
conda activate paddlepaddle
cd D:\AI\langextract-AM
git pull origin main
```

### 仅 OCR 预处理（检查 OCR 质量）

```bash
python main.py --preprocess-only --max 1
```

首次运行会自动下载 PaddleOCR-VL 1.5 模型，结束后 `AMpdf/` 目录下多出对应的 `.txt` 文件。

### 完整流程（OCR + LangExtract 抽取）

```bash
# 抽取 1 篇
python main.py --max 1 --chunk 8000

# 抽取全部
python main.py --chunk 8000

# 强制重新 OCR + 全流程
python main.py --force-ocr --max 2 --chunk 12000
```

预期日志步骤：

1. `[1/4] PaddleOCR-VL 预处理: ...`
2. `[2/4] lx.extract 分块并发 ...`
3. `[3/4] 聚合 MaterialEntity`
4. `[4/4] 转目标 JSON，过滤引用材料`

### 跳过 PaddleOCR（旧模式，直接 PyMuPDF 提文本）

```bash
python main.py --no-ocr --max 1
```

### 使用其他模型

```bash
# Gemini
python main.py --model gemini

# 直接传 OpenAI 兼容 model_id
python main.py --model ep_3nr55ube9_ernie
```

### 参数说明

| 参数 | 默认 | 说明 |
|------|------|------|
| `--model` | `env` | `gemini` / `env` / 任意 OpenAI 兼容 `model_id` |
| `--max` | 0 | 最多处理 PDF 数量，0 表示全部 |
| `--chunk` | 6000 | 分块大小（字符），单块失败会切半重试 |
| `--workers` | 1 | 分块并发数，1 为串行（便于排查） |
| `--no-ocr` | — | 跳过 PaddleOCR，直接用 PyMuPDF 提文本 |
| `--force-ocr` | — | 强制重新 OCR（即使 .txt 已存在） |
| `--preprocess-only` | — | 仅运行 OCR 预处理，不执行 LLM 抽取 |

---

## 输出格式

- **路径**：`output/he_data_{model}.jsonl`
- **格式**：每行一条 JSON，核心结构：

```jsonc
{
  "_source_pdf": "1-2024-MSEA-Ti42Hf21Nb21V16-DED.pdf",
  "role": "Target",
  "Composition_Info": {
    "Mat_ID": "M_T42",
    "Alloy_Name_Raw": "T42",
    "Formula_Normalized": "Ti42Hf21Nb21V16",       // 下标已规格化
    "Composition_JSON": "{\"Ti\": 42, \"Hf\": 21}", // 字符串格式 JSON
    "Source_DOI": "1-2024-MSEA-..."
  },
  "Process_Info": {
    "Sample_ID": "S_T42_AB",                        // AB=As-Built, HT=Heat-Treated
    "Process_Category": "AM_DED",                    // 标准化分类
    "Process_Text_For_AI": "DED. laser power 550W...",
    "Key_Params_JSON": "{\"Laser_Power_W\": 550}",
    "Main_Phase": "BCC",
    "Microstructure_Text_For_AI": "...",
    "Has_Precipitates": false,
    "Grain_Size_avg_um": 200
  },
  "Properties_Info": [
    {
      "Test_ID": "T_T42_01",
      "Sample_ID": "S_T42_AB",
      "Test_Temperature_K": 298.0,
      "Property_Type": "Yield_Strength",             // 标准化名称
      "Property_Value": 1030,
      "Property_Unit": "MPa"
    }
  ]
}
```

- `Composition_JSON` 与 `Key_Params_JSON` 为**字符串格式的 JSON**（由 `json.dumps` 生成）
- 每次运行**覆盖**该模型对应的 JSONL 文件；多篇 PDF 时按篇追加写入（线程安全）

---

## CPU 兜底（无 GPU 环境）

若在没有 GPU 的环境，需强制走 PPStructureV3：

```bash
conda activate paddlepaddle
cd D:\AI\langextract-AM

# Windows 临时设置
set OCR_ENGINE=structurev3

# 仅 OCR 预处理
python main.py --preprocess-only --max 1
```

---

## 常见问题

| 问题 | 解决方案 |
|------|----------|
| **403 访问过于频繁** | 保持 `--workers 1`，适当增大 `--chunk` 减少请求次数 |
| **某块一直卡住** | 已加单块超时（默认 480s），超时自动跳过；可通过 `CHUNK_TIMEOUT_SECONDS` 调大 |
| **JSON 解析失败** | 单块会先切半重试；思考模型常在 JSON 前输出推理内容，建议关闭思考或换模型 |
| **Connection error** | 网络断开时该块跳过，稍后重跑即可 |
| **OCR 阶段长时间无输出** | 首次会下载模型，较慢属正常；管线每 60 秒会打印 `OCR 仍在处理中...` |
| **`信息: 用提供的模式无法找到文件`** | Windows 无害提示，不影响运行 |

---

## 依赖（见 requirements.txt）

- `paddlepaddle` / `paddlepaddle-gpu`：PaddleOCR 底层框架（需单独安装）
- `paddleocr[doc-parser]`：PaddleOCR-VL 1.5 文档解析
- `langextract`：结构化抽取
- `pydantic`：数据模型与校验
- `pymupdf`：PDF 文本提取（PyMuPDF，`--no-ocr` 时使用）
- `pdfplumber`：PDF 提取备选
- `python-dotenv`：加载 `.env`

---

## License

MIT
