# MedChoice 医疗健康智能助手

> 完成时间：2026年5月 | 课程设计作业

## 项目简介

MedChoice 是一个基于 AI 的医疗健康智能助手。本工具采用 4 层 Agent 架构（安全检测→意图识别→业务分析→质量审核），结合 RAG 知识库检索，为用户提供四大核心功能。

### 核心功能

- **体检套餐选择** → 根据年龄、性别、职业、预算推荐合适的套餐
- **药品/保健品对比** → 结构化对比两种药的成分、适应症、副作用
- **体检报告解读** → 解释异常指标含义，建议就诊科室
- **就医科室推荐** → 根据症状推荐合适的挂号科室

## 快速开始

### 环境准备
```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key
cp .env.example .env
# 编辑 .env 文件，填写 DeepSeek 或 Zhipu API Key

# 3. 初始化知识库
python src/init_kb.py

# 4. 启动应用
streamlit run src/main.py
```

### 使用流程
1. 先在左侧边栏填写个人健康画像
2. 选择一个功能模块，输入问题或上传文件
3. 点击”开始分析”，等待 AI 生成结果
4. 查看结果，可进行追问或下一个问题

## 技术架构

### 架构图
```
User Input → Safety Agent → Router Agent → Business Agent → Reflection Agent → Output
                                              ↕                   ↓
                                        RAG Knowledge Base   Quality Check + Retry
```

### 技术栈

- **前端**：Streamlit 1.28+ — 深色毛玻璃主题（CSS 自定义）
- **AI 引擎**：DeepSeek / Zhipu GLM-4 — LLM API
- **知识库**：ChromaDB + BGE Embedding — 关键词+向量双模式检索（37篇文档）
- **文件解析**：PyMuPDF + 智谱 GLM-4V 云视觉 — PDF/图片解析
- **用户记忆**：JSON 文件存储 — 跨会话持久化
- **辅助框架**：Gradio 4.0+ — 备用 Web 界面

### 项目结构
```
final-project/
  .streamlit/
    config.toml          # Streamlit 深色主题配置
  src/
    main.py              # Streamlit 主页面
    app.py               # Gradio 备用界面（支持双文件上传）
    config.py            # 配置管理
    utils.py             # JSON 解析工具（3个Agent共用）
    init_kb.py           # 知识库初始化（37篇文档）
    agents/
      safety_agent.py      # 安全检测 Agent（关键词+LLM双重过滤）
      router_agent.py      # 意图识别 Agent（加权关键词+LLM）
      business_agent.py    # 业务逻辑编排 Agent（4场景+流式输出）
      reflection_agent.py  # 质量审核 Agent（5维检查+自动修正）
    tools/
      llm_tool.py          # LLM API 封装（流式 + 重试 + 云视觉OCR）
      rag_tool.py          # RAG 检索（关键词+嵌入双模式）
      doc_parser.py        # 文件解析（PDF/TXT/图片，含Tesseract降级）
    memory/
      user_memory.py       # 用户画像持久化
  tests/
    test_cases.py         # 5 个单元测试
    integration_test.py   # 6 个集成测试
  docs/
    test-cases.md         # 测试文档
    答辩PPT大纲.md        # 答辩PPT大纲
  data/
    chroma_db/            # 向量数据库
    models/               # BGE 嵌入模型缓存
    temp/                 # 上传文件临时存储
    user_profiles/        # 用户画像 JSON 文件
  screenshots/            # 运行截图（11张，覆盖全部场景+亮点）
  requirements.txt
  README.md
```

## 测试说明

```bash
# 单元测试（5个）
python tests/test_cases.py

# 集成测试（6个，需要 API Key）
python tests/integration_test.py
```

包含 11 个测试用例（5 单元 + 6 集成）：
- TC-01: 体检套餐推荐路由测试
- TC-02: 药品对比输出测试
- TC-03: 体检报告解读安全性测试
- TC-04: 安全拦截功能测试
- TC-05: 异常处理测试
- T01-T06: 端到端集成测试（全链路 + OCR + 安全拦截）

## 项目亮点

1. **4层 Agent 架构**：安全检测→意图识别→业务分析→质量审核，层层过滤
2. **RAG 知识库**：支持向量检索和关键词检索双模式
3. **多轮对话**：记住历史对话，支持追问
4. **个性化推荐**：基于用户画像的精准匹配
5. **多模态输入**：支持文字/图片/PDF 多种输入
6. **质量审核**：Reflection Agent 自动检查输出质量

## 运行截图

`screenshots/` 目录包含 11 张截图，覆盖全部功能：

| 文件 | 内容 |
|------|------|
| `01-home.png` | 首页全貌 — 4 标签 / AI 状态灯 / 免责声明 |
| `02-profile-saved.png` | 个人画像保存 — 时间戳 "✅ 画像已加载" |
| `03-physical-exam-streaming.png` | 流式输出过程 — AI 逐字生成中 |
| `04-physical-exam-result.png` | 体检套餐完整结果 — 含反思自检标识 |
| `05-drug-compare-text.png` | 药品对比结构化表格 |
| `06-drug-compare-upload.png` | 双文件上传 OCR + 药品对比 |
| `07-report-reading-ocr.png` | 体检报告图片上传 OCR + 解读 |
| `08-department.png` | 就医科室推荐 |
| `09-safety-rejection.png` | 安全拦截 — 危险内容拒绝 + 心理援助热线 |
| `10-multi-turn.png` | 多轮对话 — 追问上下文记忆 |
| `11-chat-history.png` | 对话历史面板 — 展开显示历史记录 |

## 免责声明
> 本工具为课程设计作业，所有分析结果仅供参考，**不能替代医生诊断**。如有不适，请及时就医。