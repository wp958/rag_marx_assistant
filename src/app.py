"""
Capital RAG Web Application with Qwen (Aliyun Bailian)
"""

import json
import os
from flask import Flask, render_template, request, jsonify
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI


# ============================================================
# CONFIG
# ============================================================

# Chunks and database paths
CHUNKS_FILE = r"D:\zhuo mian\rag\src\rag_chunks\chunks.json"
DB_DIR = r"D:\zhuo mian\rag\src\vector_db"
EMBEDDING_MODEL = "shibing624/text2vec-base-chinese"
TOP_K = 5

# Aliyun Bailian (Qwen) API - FILL IN YOUR API KEY
QWEN_API_KEY = "sk-90f016edb0ed4c99b159efda69774ae7"  # 把这里换成你的 API Key
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL = "qwen-plus"  # 可选: qwen-turbo, qwen-plus, qwen-max

# ============================================================


app = Flask(__name__)

# Global variables
model = None
collection = None
qwen_client = None


def init_qwen():
    """Initialize Qwen client"""
    global qwen_client
    if QWEN_API_KEY and QWEN_API_KEY != "sk-xxxxx":
        qwen_client = OpenAI(
            api_key=QWEN_API_KEY,
            base_url=QWEN_BASE_URL,
        )
        print(f"  Qwen AI configured: {QWEN_MODEL}")
    else:
        qwen_client = None
        print("  [!] Qwen API Key not configured")


def call_qwen(question, context):
    """
    Call Qwen to generate an answer based on context
    """
    if not qwen_client:
        return None

    system_prompt = """你是一个专业的马克思主义研究助手。请根据提供的《资本论》原文段落，回答用户的问题。

要求：
1. 回答必须基于提供的原文内容
2. 用通俗易懂的语言解释
3. 如果原文中没有相关内容，请诚实说明
4. 回答控制在300字以内"""

    user_prompt = f"""问题：{question}

以下是《资本论》中的相关段落：

{context}

请根据以上原文回答问题。"""

    try:
        response = qwen_client.chat.completions.create(
            model=QWEN_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=1024,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Qwen API error: {e}")
        return None


# ============================================================
# RAG FUNCTIONS
# ============================================================

def load_chunks():
    print("Loading chunks...")
    with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    print(f"  Loaded {len(chunks)} chunks")
    return chunks


def build_database(chunks):
    global model, collection

    print("\n" + "=" * 50)
    print("Building vector database...")
    print("=" * 50)

    print("\n[1/3] Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("  Done!")

    print("\n[2/3] Creating database...")
    os.makedirs(DB_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=DB_DIR)

    try:
        client.delete_collection("capital")
    except:
        pass

    collection = client.create_collection(name="capital")

    total = len(chunks)
    print(f"\n[3/3] Adding {total} chunks...")

    batch_size = 100
    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]
        ids = [str(item['id']) for item in batch]
        documents = [item['content'] for item in batch]
        embeddings = model.encode(documents).tolist()
        collection.add(ids=ids, documents=documents, embeddings=embeddings)
        done = min(i + batch_size, total)
        print(f"  Progress: {done}/{total} ({done*100//total}%)")

    print("\n  Database ready!")


def load_database():
    global model, collection
    print("Loading existing database...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection("capital")
    print(f"  Loaded! {collection.count()} chunks")


def init():
    db_exists = os.path.exists(DB_DIR) and os.listdir(DB_DIR)
    if db_exists:
        load_database()
    else:
        chunks = load_chunks()
        build_database(chunks)

    init_qwen()


def search(question):
    question_embedding = model.encode([question]).tolist()
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=TOP_K,
    )
    return results


# ============================================================
# WEB ROUTES
# ============================================================

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get('question', '').strip()

    if not question:
        return jsonify({'error': 'Please enter a question'})

    # Step 1: Search for relevant chunks
    results = search(question)
    documents = results['documents'][0]
    distances = results['distances'][0]

    # Step 2: Format references
    references = []
    context_parts = []
    for i, (doc, dist) in enumerate(zip(documents, distances)):
        similarity = max(0, (1 - dist) * 100)
        references.append({
            'id': i + 1,
            'content': doc,
            'relevance': round(similarity, 1)
        })
        context_parts.append(f"[段落{i+1}]\n{doc}")

    context = "\n\n".join(context_parts)

    # Step 3: Generate AI answer
    ai_answer = call_qwen(question, context)

    return jsonify({
        'question': question,
        'ai_answer': ai_answer,
        'references': references
    })


if __name__ == '__main__':
    print("=" * 50)
    print("  Das Kapital RAG System")
    print("=" * 50)

    init()

    print("\n" + "=" * 50)
    print("  Open in browser: http://127.0.0.1:5000")
    print("  Press Ctrl+C to stop")
    print("=" * 50 + "\n")

    app.run(debug=False, host='127.0.0.1', port=5000)