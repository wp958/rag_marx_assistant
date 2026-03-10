"""
Capital RAG Q&A System
Features:
  1. Original capital.txt knowledge base
  2. Multi-file upload (PDF/TXT/DOCX/MD)
  3. Multi-turn conversation with context memory
  4. Multiple prompt strategies (Normal/Deep/Concise)
"""

import json
import os
import uuid
from datetime import datetime

from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ============================================================
# CONFIG
# ============================================================
CHUNKS_FILE = r"D:\zhuo mian\rag\src\rag_chunks\chunks.json"
DB_DIR = r"D:\zhuo mian\rag\src\vector_db"
UPLOAD_DIR = r"D:\zhuo mian\rag\uploads"
EMBEDDING_MODEL = "shibing624/text2vec-base-chinese"
TOP_K = 5

QWEN_API_KEY = "your-api-key-here"
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL = "qwen-plus"

ALLOWED_EXTENSIONS = {'pdf', 'txt', 'docx', 'md'}
MAX_HISTORY = 10

# ============================================================
# [NEW] PROMPT STRATEGIES
# ============================================================
PROMPT_STRATEGIES = {
    "normal": {
        "name": "Normal Mode",
        "label": "standard",
        "temperature": 0.7,
        "system_prompt": """你是一个专业的知识库问答助手。请根据提供的文档内容回答用户的问题。

要求：
1. 回答必须基于提供的文档内容，不要编造
2. 用通俗易懂的语言解释
3. 适当引用原文关键句子作为支撑
4. 如果文档中没有相关内容，请诚实说明
5. 回答控制在300字以内
6. 如果用户是追问，结合之前的对话上下文回答""",

        "user_template": "问题：{question}\n\n以下是相关文档内容：\n{context}\n\n请根据以上内容回答问题。"
    },

    "deep": {
        "name": "Deep Analysis Mode",
        "label": "deep",
        "temperature": 0.5,
        "system_prompt": """你是一个严谨的学术研究助手，擅长深度分析文本内容。请根据提供的文档内容，对用户的问题进行深入、系统的分析。

要求：
1. 必须基于提供的原文内容，严禁编造
2. 分析要有层次，使用"首先、其次、最后"或"第一、第二、第三"等结构
3. 关键论点必须引用原文原句，用引号标注
4. 如果不同段落有不同观点，要对比分析
5. 在回答末尾总结核心要点
6. 如果原文信息不足，明确指出哪些方面缺少依据
7. 如果用户是追问，结合之前的对话上下文深入展开

回答格式：
【分析】
（分层论述）

【原文引用】
（关键原句）

【总结】
（核心要点）""",

        "user_template": """请对以下问题进行深度分析。

问题：{question}

参考文档内容：
{context}

请严格基于以上文档内容进行深入分析，引用原文关键语句。"""
    },

    "concise": {
        "name": "Concise Mode",
        "label": "concise",
        "temperature": 0.3,
        "system_prompt": """你是一个高效的问答助手。请用最简洁的语言回答问题。

要求：
1. 直接给出答案，不要铺垫
2. 控制在100字以内
3. 使用短句
4. 只基于提供的文档内容回答
5. 如果文档中没有答案，直接说"文档中未涉及此内容"
6. 如果用户是追问，简洁回答即可""",

        "user_template": "问题：{question}\n\n文档内容：\n{context}\n\n请简洁回答。"
    }
}
# ============================================================

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.secret_key = 'rag-capital-2024'

# Global
model = None
collection = None
llm_client = None
conversations = {}


# ============================================================
# INIT
# ============================================================

def init_model():
    global model
    print("[1/3] Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("      Done!")


def init_db():
    global collection
    os.makedirs(DB_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=DB_DIR)

    if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
        try:
            collection = client.get_collection("capital")
            print("[2/3] Loaded existing database: %d chunks" % collection.count())
            return
        except:
            pass

    print("[2/3] Building database from chunks.json...")
    with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    collection = client.create_collection("capital")

    batch_size = 100
    total = len(chunks)
    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]
        ids = [str(item['id']) for item in batch]
        docs = [item['content'] for item in batch]
        embs = model.encode(docs).tolist()
        collection.add(ids=ids, documents=docs, embeddings=embs)
        done = min(i + batch_size, total)
        print("      %d/%d (%.0f%%)" % (done, total, done / total * 100))

    print("      Done! %d chunks" % collection.count())


def init_llm():
    global llm_client
    if QWEN_API_KEY and QWEN_API_KEY != "your-api-key-here":
        llm_client = OpenAI(api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)
        print("[3/3] LLM ready: %s" % QWEN_MODEL)
    else:
        llm_client = None
        print("[3/3] LLM not configured")


def init():
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    init_model()
    init_db()
    init_llm()


# ============================================================
# CONVERSATION HISTORY
# ============================================================

def get_session_id():
    if 'sid' not in session:
        session['sid'] = str(uuid.uuid4())
    return session['sid']


def get_history(session_id):
    if session_id not in conversations:
        conversations[session_id] = []
    return conversations[session_id]


def add_to_history(session_id, role, content):
    history = get_history(session_id)
    history.append({
        'role': role,
        'content': content,
        'timestamp': datetime.now().strftime('%H:%M:%S')
    })
    if len(history) > MAX_HISTORY * 2:
        conversations[session_id] = history[-(MAX_HISTORY * 2):]


def clear_history(session_id):
    conversations[session_id] = []


def build_query_with_context(question, session_id):
    history = get_history(session_id)
    if not history:
        return question
    recent = history[-(4):]
    context_parts = []
    for msg in recent:
        if msg['role'] == 'user':
            context_parts.append(msg['content'])
    if context_parts:
        enhanced = " ".join(context_parts[-2:]) + " " + question
        return enhanced
    return question


# ============================================================
# UPLOAD & INDEX
# ============================================================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def add_to_index(chunks_data):
    if not chunks_data:
        return 0
    current_count = collection.count()
    ids = []
    docs = []
    metas = []
    for i, chunk in enumerate(chunks_data):
        ids.append("upload_%d" % (current_count + i))
        docs.append(chunk['content'])
        metas.append(chunk.get('metadata', {}))
    embs = model.encode(docs).tolist()
    collection.add(ids=ids, documents=docs, embeddings=embs, metadatas=metas)
    return len(ids)


# ============================================================
# SEARCH & GENERATE
# ============================================================

def search(question, top_k=TOP_K):
    emb = model.encode([question]).tolist()
    results = collection.query(query_embeddings=emb, n_results=top_k)
    return results


def generate(question, contexts, session_id, strategy_key="normal"):
    """[NEW] Generate answer using selected prompt strategy"""
    if not llm_client:
        return None

    # [NEW] Get the selected strategy
    strategy = PROMPT_STRATEGIES.get(strategy_key, PROMPT_STRATEGIES["normal"])

    # Build context
    context_text = ""
    for i, doc in enumerate(contexts):
        context_text += "\n[段落%d]\n%s\n" % (i + 1, doc)

    # [NEW] Use strategy-specific template
    user_prompt = strategy["user_template"].format(
        question=question,
        context=context_text
    )

    # Build messages with history
    messages = [{"role": "system", "content": strategy["system_prompt"]}]

    history = get_history(session_id)
    for msg in history[-(MAX_HISTORY * 2):]:
        messages.append({
            "role": msg['role'],
            "content": msg['content']
        })

    messages.append({"role": "user", "content": user_prompt})

    try:
        resp = llm_client.chat.completions.create(
            model=QWEN_MODEL,
            messages=messages,
            max_tokens=1024,
            temperature=strategy["temperature"]  # [NEW] strategy-specific temperature
        )
        return resp.choices[0].message.content
    except Exception as e:
        print("LLM error: %s" % e)
        return None


# ============================================================
# ROUTES
# ============================================================

@app.route('/')
def home():
    return render_template('index.html')


# [NEW] Return available strategies
@app.route('/strategies')
def strategies():
    result = {}
    for key, val in PROMPT_STRATEGIES.items():
        result[key] = {
            'name': val['name'],
            'label': val['label']
        }
    return jsonify(result)


@app.route('/upload', methods=['POST'])
def upload():
    from parsers import parse_file, chunk_text

    if 'files' not in request.files:
        return jsonify({'error': 'no files'}), 400

    files = request.files.getlist('files')
    results = []

    for file in files:
        if file.filename == '' or not allowed_file(file.filename):
            results.append({
                'filename': file.filename or 'unknown',
                'status': 'error',
                'message': 'unsupported file type'
            })
            continue

        try:
            filename = secure_filename(file.filename)
            if not filename or filename == '':
                filename = file.filename
            filepath = os.path.join(UPLOAD_DIR, filename)
            file.save(filepath)

            doc = parse_file(filepath)
            chunks = chunk_text(doc['content'])

            chunks_data = []
            for j, c in enumerate(chunks):
                chunks_data.append({
                    'content': c,
                    'metadata': {
                        'filename': doc['filename'],
                        'chunk_index': j,
                        'file_type': doc['file_type']
                    }
                })

            added = add_to_index(chunks_data)
            results.append({
                'filename': doc['filename'],
                'status': 'success',
                'chars': doc['chars'],
                'chunks': added
            })

        except Exception as e:
            results.append({
                'filename': file.filename,
                'status': 'error',
                'message': str(e)
            })

    return jsonify({
        'results': results,
        'total_chunks': collection.count()
    })


@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get('question', '').strip()
    strategy_key = data.get('strategy', 'normal')  # [NEW] get strategy from request

    if not question:
        return jsonify({'error': 'empty question'})

    session_id = get_session_id()
    enhanced_query = build_query_with_context(question, session_id)

    results = search(enhanced_query)
    documents = results['documents'][0]
    distances = results['distances'][0]
    metadatas = results.get('metadatas', [[{}] * len(documents)])[0]

    references = []
    for i, (doc, dist, meta) in enumerate(zip(documents, distances, metadatas)):
        sim = max(0, (1 - dist) * 100)
        references.append({
            'id': i + 1,
            'content': doc,
            'relevance': round(sim, 1),
            'filename': meta.get('filename', 'capital.txt') if meta else 'capital.txt'
        })

    # [NEW] Pass strategy to generate
    ai_answer = generate(question, documents, session_id, strategy_key)

    add_to_history(session_id, 'user', question)
    if ai_answer:
        add_to_history(session_id, 'assistant', ai_answer)

    # [NEW] Return which strategy was used
    strategy = PROMPT_STRATEGIES.get(strategy_key, PROMPT_STRATEGIES["normal"])

    return jsonify({
        'question': question,
        'ai_answer': ai_answer,
        'references': references,
        'strategy_used': strategy['label']
    })


@app.route('/clear_chat', methods=['POST'])
def clear_chat():
    session_id = get_session_id()
    clear_history(session_id)
    return jsonify({'status': 'ok'})


@app.route('/stats')
def stats():
    file_count = len(os.listdir(UPLOAD_DIR)) if os.path.exists(UPLOAD_DIR) else 0
    return jsonify({
        'total_chunks': collection.count(),
        'uploaded_files': file_count
    })


@app.route('/clear_uploads', methods=['POST'])
def clear_uploads():
    if os.path.exists(UPLOAD_DIR):
        for f in os.listdir(UPLOAD_DIR):
            os.remove(os.path.join(UPLOAD_DIR, f))
    init_db()
    return jsonify({
        'status': 'ok',
        'total_chunks': collection.count()
    })


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("=" * 50)
    print("  Capital RAG Q&A System")
    print("  Upload / Chat / Prompt Strategies")
    print("=" * 50)

    # [NEW] Show available strategies
    print("\n  Prompt Strategies:")
    for key, val in PROMPT_STRATEGIES.items():
        print("    [%s] %s (temp=%.1f)" % (key, val['name'], val['temperature']))

    init()

    print("\n" + "=" * 50)
    print("  http://127.0.0.1:5000")
    print("  Ctrl+C to stop")
    print("=" * 50 + "\n")

    app.run(debug=False, host='127.0.0.1', port=5000)
