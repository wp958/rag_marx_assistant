# 资本论 RAG 智能问答系统

基于马克思《资本论》全三卷构建的 RAG（检索增强生成）问答系统。自动检索相关原文段落，通过 Qwen-Plus 大模型生成精准回答。

## 系统架构

用户提问 -> Query Embedding (text2vec-base-chinese) -> Vector Search (ChromaDB Top-5) -> Context + Question -> Qwen-Plus API -> AI回答 + 原文参考段落

## 核心技术

| 组件 | 选型 | 说明 |
|------|------|------|
| 数据采集 | BeautifulSoup | 爬取 marxists.org 全三卷 约160万字 |
| 文本分块 | 递归字符分割 | 500字每块 100字重叠 句子级去重 |
| Embedding | text2vec-base-chinese | 中文语义编码 768维向量 |
| 向量数据库 | ChromaDB | 轻量持久化 余弦相似度检索 |
| 大模型 | Qwen-Plus | OpenAI兼容API 基于检索上下文生成回答 |
| Web | Flask | 响应式界面 实时问答 |

## 项目结构

- data/capital.txt - 资本论全文 约160万字
- src/scraper.py - 数据采集 三卷爬虫
- src/build_index.py - 文本清洗 分块 向量索引构建
- src/app.py - RAG Pipeline 加 Flask Web服务
- src/templates/index.html - 前端界面
- requirements.txt - Python依赖

## 快速开始

### 安装

git clone https://github.com/YOUR_USERNAME/rag_marx_assistant.git

cd rag_marx_assistant

pip install -r requirements.txt

### 采集数据

python src/scraper.py

### 构建索引

python src/build_index.py

### 启动服务

Windows CMD设置API Key: set QWEN_API_KEY=sk-your-key-here

启动: python src/app.py

浏览器打开 http://127.0.0.1:5000

## 实现细节

### 数据处理

针对 marxists.org 的HTML结构定制解析。基于hr标签精确切除导航栏，检测锚点链接自动移除章内目录。三卷URL模式不同，第一卷使用NN.htm格式，二三卷使用 /marx-engels/24 和 /marx-engels/25 路径，分别处理。句子级去重消除页面内目录与正文的重复内容。

### 分块策略

递归分割优先级: 段落 然后 换行 然后 句号 然后 问号 然后 分号 然后 逗号。500字每块加100字重叠，平衡检索精度与上下文完整性。最终生成约3500个高质量文本块。

### RAG Pipeline

Retrieve阶段: 问题向量化后在ChromaDB中检索Top5相似块。Generate阶段: System Prompt约束模型只基于原文回答，抑制幻觉。Temperature设为0.7平衡准确性与流畅度。

### 性能指标

| 指标 | 数值 |
|------|------|
| 数据规模 | 约160万字 约3500 chunks |
| 检索延迟 | 小于 100ms |
| 端到端延迟 | 2到5秒 含LLM生成 |

## 后续优化方向

- 添加 Reranker 重排序提升检索精度
- Hybrid Search 语义加BM25关键词检索
- 多轮对话支持
- Chunk Metadata 按卷号章节过滤检索
- Docker 部署

## License

MIT