# 资本论 RAG 智能问答系统

> 基于检索增强生成（RAG）技术，对马克思《资本论》全三卷进行智能问答

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.4-FF6F61?style=for-the-badge)
![LangChain](https://img.shields.io/badge/LangChain-Splitter-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![Qwen](https://img.shields.io/badge/Qwen--Plus-LLM-6366F1?style=for-the-badge)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)

![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-success?style=flat-square)
![RAG](https://img.shields.io/badge/RAG-Retrieval--Augmented--Generation-blue?style=flat-square)

---

## 项目亮点

| 能力 | 实现 |
|:---:|:---|
| 📥 **数据采集** | 自动爬取《资本论》全三卷，160万字，3500+文本块 |
| 📁 **多格式上传** | 支持 PDF / TXT / DOCX / MD 批量上传，自动解析入库 |
| 🔍 **语义检索** | text2vec 中文向量化 + ChromaDB 毫秒级检索 |
| 💬 **多轮对话** | 上下文记忆管理，支持连续追问 |
| 🎯 **Prompt策略** | 3种策略可切换（标准/深度分析/简洁），回答质量可控 |
| 🤖 **智能回答** | Qwen-Plus 大模型基于原文生成回答，拒绝幻觉 |

---

## 系统架构
text

                ┌─────────────────────────────────────────┐
                │            用户提问                      │
                │       "什么是剩余价值？"                  │
                └──────────────┬──────────────────────────┘
                               │
                ┌──────────────┴──────────────┐
                │     Query Context Enhancement │
                │  结合对话历史增强检索query      │
                └──────────────┬──────────────┘
                               │
┌──────────────────────────────────┴───────────────────────────────────┐
│ RETRIEVAL 检索层 │
│ ┌────────────────┐ ┌─────────────────┐ ┌──────────────────┐ │
│ │ Query Encoding │───▶│ text2vec-chinese │───▶│ ChromaDB Top-5 │ │
│ │ 问题向量化 │ │ 768维语义向量 │ │ 余弦相似检索 │ │
│ └────────────────┘ └─────────────────┘ └────────┬─────────┘ │
└────────────────────────────────────────────────────────┼─────────────┘
│
┌────────────────────────────────────┘
│
┌───────────────────┴─────────────────────────────────────────────────┐
│ GENERATION 生成层 │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Prompt Strategy Selection │ │
│ │ ┌─────────┐ ┌──────────────┐ ┌────────────┐ │ │
│ │ │ Standard │ │ Deep Analysis│ │ Concise │ │ │
│ │ │ temp=0.7 │ │ temp=0.5 │ │ temp=0.3 │ │ │
│ │ └─────────┘ └──────────────┘ └────────────┘ │ │
│ └──────────────────────┬──────────────────────────────────────┘ │
│ │ │
│ ┌──────────────────────┴──────────────────────────────────────┐ │
│ │ System Prompt + Conversation History + Context + Question │ │
│ └──────────────────────┬──────────────────────────────────────┘ │
│ │ │
│ ▼ │
│ ┌─────────────────────┐ │
│ │ Qwen-Plus API │ │
│ └──────────┬──────────┘ │
└─────────────────────────┼───────────────────────────────────────────┘
│
▼
┌─────────────────────┐
│ AI回答 + 原文引用 │
└─────────────────────┘

text


---

## 技术选型

<table>
<tr>
<td align="center" width="96">
<img src="https://skillicons.dev/icons?i=python" width="48" height="48" alt="Python" />
<br>Python
</td>
<td align="center" width="96">
<img src="https://skillicons.dev/icons?i=flask" width="48" height="48" alt="Flask" />
<br>Flask
</td>
<td align="center" width="96">
<img src="https://raw.githubusercontent.com/langchain-ai/langchain/master/docs/static/img/brand/wordmark.png" width="48" height="48" alt="LangChain" />
<br>LangChain
</td>
<td align="center" width="96">
<img src="https://huggingface.co/front/assets/huggingface_logo.svg" width="48" height="48" alt="HuggingFace" />
<br>HuggingFace
</td>
<td align="center" width="96">
<img src="https://www.trychroma.com/chroma-logo.png" width="48" height="48" alt="ChromaDB" />
<br>ChromaDB
</td>
<td align="center" width="96">
<img src="https://skillicons.dev/icons?i=html" width="48" height="48" alt="HTML" />
<br>HTML/CSS
</td>
</tr>
</table>

| 模块 | 技术 | 选型理由 |
|:---|:---|:---|
| **数据采集** | `BeautifulSoup` + `Requests` | 灵活解析HTML，处理三卷不同URL结构 |
| **文件解析** | `PyPDF2` + `python-docx` + `chardet` | 多格式支持，自动编码检测 |
| **文本分块** | `递归字符分割` | 500字/块 + 100字重叠，保留语义完整性 |
| **向量编码** | `text2vec-base-chinese` | 专为中文优化，768维语义向量 |
| **向量数据库** | `ChromaDB` | 轻量持久化，毫秒级余弦相似检索 |
| **大语言模型** | `Qwen-Plus` | OpenAI兼容API，中文理解能力强 |
| **Web框架** | `Flask` | 轻量灵活，前后端分离 |

---

## 项目结构
rag_marx_assistant/
│
├── data/
│ └── capital.txt # 《资本论》全文 (160万字)
│
├── uploads/ # 用户上传文件存储
│
├── src/
│ ├── scraper.py # 数据采集：三卷爬虫
│ ├── build_index.py # 文本清洗 + 分块 + 向量索引构建
│ ├── parsers.py # 多格式文件解析器 (PDF/TXT/DOCX)
│ ├── app.py # RAG Pipeline + Web服务 + 对话管理
│ └── templates/
│ └── index.html # 前端界面
│
├── requirements.txt
├── README.md
└── LICENSE

text


---

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/wp958/rag_marx_assistant.git
cd rag_marx_assistant
2. 安装依赖
Bash

pip install -r requirements.txt
3. 采集数据
Bash

python src/scraper.py
4. 构建索引
Bash

python src/build_index.py
5. 启动服务
Bash

# Windows
set QWEN_API_KEY=sk-your-key-here
python src/app.py
6. 打开浏览器
访问 http://127.0.0.1:5000

核心实现
数据处理 Pipeline
text

原始HTML → 导航栏去除 → 正文提取 → 句子级去重 → 递归分块 → JSON存储
基于 <hr> 标签精确切除页面导航
检测 #锚点 链接自动移除章内目录
三卷URL模式不同，分别定制解析
多文件上传处理
text

用户上传文件 → 格式检测 → 对应解析器 → 文本提取 → 分块 → 向量化 → 加入索引
PDF: PyPDF2 逐页提取
DOCX: python-docx 段落提取
TXT/MD: chardet 自动编码检测
多轮对话管理
text

用户追问 → 提取历史上下文 → 增强检索Query → 检索 → 历史+上下文+问题 → LLM生成
Session 级别对话隔离
历史窗口控制（最近10轮）
Query Enhancement：结合历史问题优化检索
Prompt 策略工程
策略	Temperature	特点	适用场景
标准模式	0.7	平衡准确与流畅，300字以内	一般性问题
深度分析	0.5	分层论述 + 原文引用 + 总结	学术研究
简洁模式	0.3	直接回答，100字以内	快速查询
关键设计：

System Prompt 约束模型只基于原文回答，抑制幻觉
不同策略使用不同 Temperature 控制输出风格
User Prompt Template 针对不同策略定制结构
性能指标
指标	数值
数据规模	160万字 / 3,500 chunks
向量维度	768维
检索延迟	< 100ms
端到端延迟	2-5秒（含LLM生成）
支持文件格式	PDF / TXT / DOCX / MD
对话记忆	最近10轮
Prompt策略	3种可切换
优化方向
 Reranker重排序：引入Cross-Encoder提升检索精度
 Hybrid Search：语义检索 + BM25关键词检索融合
 元数据过滤：按卷号/章节/文件名精准定位
 流式输出：SSE实现打字机效果
 Docker部署：容器化一键部署
 用户反馈：采集用户评价持续优化Prompt
技术栈总览
text

┌──────────────────────────────────────────────────────────┐
│                     Frontend                              │
│                HTML / CSS / JavaScript                    │
└────────────────────────┬─────────────────────────────────┘
                         │
┌────────────────────────┴─────────────────────────────────┐
│                     Backend                               │
│              Flask + Session Management                   │
└────────────────────────┬─────────────────────────────────┘
                         │
┌────────────────────────┴─────────────────────────────────┐
│                   RAG Pipeline                            │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌─────────┐  │
│  │ File      │ │ Embedding │ │ VectorDB  │ │ LLM     │  │
│  │ Parsers   │ │ text2vec  │ │ ChromaDB  │ │ Qwen+   │  │
│  └───────────┘ └───────────┘ └───────────┘ └─────────┘  │
│                                                           │
│  ┌───────────────────────────────────────────────────┐   │
│  │            Prompt Strategy Engine                  │   │
│  │     Standard  /  Deep Analysis  /  Concise        │   │
│  └───────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
License
MIT License

作者
GitHub: @wp958

如有问题欢迎提 Issue！
