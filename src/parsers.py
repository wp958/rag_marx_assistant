"""
文件解析器
支持 PDF / TXT / MD / DOCX 四种格式
"""

import os
import re
import chardet
from PyPDF2 import PdfReader
from docx import Document


def parse_file(file_path):
    """
    根据文件类型自动选择解析方式

    Args:
        file_path: 文件完整路径

    Returns:
        dict: {filename, content, chars, file_type}
    """
    ext = file_path.lower().rsplit('.', 1)[-1]

    if ext == 'pdf':
        return parse_pdf(file_path)
    elif ext in ('txt', 'md'):
        return parse_txt(file_path)
    elif ext == 'docx':
        return parse_docx(file_path)
    else:
        raise ValueError("not supported: " + ext)


def parse_pdf(file_path):
    """解析PDF文件"""
    reader = PdfReader(file_path)
    parts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            parts.append(text.strip())

    content = "\n\n".join(parts)
    return {
        'filename': os.path.basename(file_path),
        'content': content,
        'chars': len(content),
        'file_type': 'pdf'
    }


def parse_txt(file_path):
    """解析TXT/MD文件，自动检测编码"""
    with open(file_path, 'rb') as f:
        raw = f.read()
        detected = chardet.detect(raw)
        encoding = detected['encoding'] or 'utf-8'

    with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
        content = f.read().strip()

    return {
        'filename': os.path.basename(file_path),
        'content': content,
        'chars': len(content),
        'file_type': 'txt'
    }


def parse_docx(file_path):
    """解析Word DOCX文件"""
    doc = Document(file_path)
    parts = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            parts.append(text)

    content = "\n\n".join(parts)
    return {
        'filename': os.path.basename(file_path),
        'content': content,
        'chars': len(content),
        'file_type': 'docx'
    }


def chunk_text(text, chunk_size=500, overlap=100):
    """
    将长文本切分为小块

    Args:
        text: 原始文本
        chunk_size: 每块大小
        overlap: 重叠字符数

    Returns:
        list: 文本块列表
    """
    text = re.sub(r'\n{3,}', '\n\n', text).strip()

    if len(text) <= chunk_size:
        return [text] if text else []

    separators = ['\n\n', '\n', '。', '！', '？', '；', '，', ' ', '']

    best_sep = ''
    for sep in separators:
        if sep in text:
            best_sep = sep
            break

    if best_sep:
        parts = text.split(best_sep)
    else:
        parts = list(text)

    chunks = []
    current = ""

    for part in parts:
        test = current + best_sep + part if current else part
        if len(test) <= chunk_size:
            current = test
        else:
            if current.strip():
                chunks.append(current.strip())
            if len(part) > chunk_size:
                sub = chunk_text(part, chunk_size, overlap)
                chunks.extend(sub)
                current = ""
            else:
                current = part

    if current.strip():
        chunks.append(current.strip())

    # add overlap
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            ov = prev[-overlap:] if len(prev) > overlap else prev
            for char in ['。', '！', '？', '\n']:
                pos = ov.find(char)
                if pos != -1:
                    ov = ov[pos + 1:]
                    break
            new = ov.strip() + ' ' + chunks[i]
            overlapped.append(new.strip())
        chunks = overlapped

    chunks = [c for c in chunks if len(c) >= 20]
    return chunks