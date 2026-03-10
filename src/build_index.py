"""
RAG Text Chunker
Turn Capital full text into chunks ready for vector database

What is chunking?
=================
Imagine you have a 1000-page book.
If someone asks a question, you don't want to search the entire book.
You want to search small "chunks" (like paragraphs or sections).

This script cuts the full text into small pieces:
- Each piece is about 500 characters (roughly a paragraph)
- Adjacent pieces overlap by 100 characters (so no sentence is cut in half)

Input:  FULL_Capital_All_Volumes.txt (one big file)
Output: chunks.json (many small pieces, ready for embedding)
"""

import os
import re
import json

# ============================================================
# CONFIG - you can adjust these numbers
# ============================================================

# where your merged text file is
# 改成你的真实路径（r 表示原始字符串，避免反斜杠问题）
INPUT_FILE = r"D:\zhuo mian\rag\data\capital_texts_final\FULL_Capital_All_Volumes.txt"

# where to save the chunks
OUTPUT_DIR = "rag_chunks"

# chunk size (in characters, not words)
# 500 is a good default for Chinese text
# too small (< 200) = loses context
# too big (> 1000) = search results too broad
CHUNK_SIZE = 500

# overlap between adjacent chunks
# this prevents sentences from being cut in the middle
# usually 10-20% of chunk_size
CHUNK_OVERLAP = 100


# ============================================================


def clean_text(text):
    """
    Clean up the raw text before chunking
    Fixed: sentence-level deduplication
    """
    # Step 1: remove separator lines
    text = re.sub(r'={3,}', '', text)

    # Step 2: remove footnote markers like [1] [2] [3]
    # they cause duplicate detection problems
    text = re.sub(r'\[\d+\]', '', text)

    # Step 3: split entire text into sentences
    # Chinese sentences end with these punctuation marks
    sentences = re.split(r'((?<=[。！？\n])\s*)', text)

    # rejoin into clean sentences
    clean_sentences = []
    current = ''
    for part in sentences:
        current += part
        # if current piece ends with sentence-ending punctuation or newline
        if re.search(r'[。！？]\s*$', current) or current.endswith('\n'):
            stripped = current.strip()
            if stripped:
                clean_sentences.append(stripped)
            current = ''
    if current.strip():
        clean_sentences.append(current.strip())

    # Step 4: remove duplicate sentences (key fix!)
    seen = set()
    unique_sentences = []
    for sentence in clean_sentences:
        # normalize: remove all whitespace for comparison
        normalized = re.sub(r'\s+', '', sentence)

        # skip very short strings (punctuation, numbers, etc)
        if len(normalized) < 15:
            unique_sentences.append(sentence)
            continue

        if normalized in seen:
            continue  # skip duplicate

        seen.add(normalized)
        unique_sentences.append(sentence)

    text = '\n'.join(unique_sentences)

    # Step 5: remove single-character lines (noise)
    lines = text.split('\n')
    lines = [l.strip() for l in lines if len(l.strip()) > 1 or l.strip() == '']
    text = '\n'.join(lines)

    # Step 6: collapse multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def split_into_chapters(text):
    """
    First split by chapter (using the ===== markers or title patterns)
    This helps us tag each chunk with its source chapter
    """
    # split by the chapter markers we created during scraping
    # pattern: empty line, then a line that looks like a title
    chapters = []

    # try to split by our merge markers
    parts = re.split(r'\n\s*\n(?=.{0,10}(?:chapter|Chapter|\x00|Volume|volume))', text)

    # if that didn't work well, just use the whole text
    if len(parts) <= 1:
        chapters.append({
            'title': 'Capital',
            'text': text
        })
    else:
        for part in parts:
            lines = part.strip().split('\n')
            title = lines[0] if lines else 'Unknown'
            chapters.append({
                'title': title[:100],
                'text': part.strip()
            })

    return chapters


def recursive_split(text, chunk_size, chunk_overlap):
    """
    Split text into chunks, trying to break at natural boundaries

    How it works (step by step):

    1. Try to split at paragraph boundaries (double newline)
    2. If a paragraph is still too long, split at sentence boundaries
    3. If a sentence is still too long, split at comma boundaries
    4. Last resort: split at any position

    This is the same algorithm as LangChain's RecursiveCharacterTextSplitter
    but written from scratch so you don't need extra dependencies
    """
    # separators in order of preference
    separators = [
        '\n\n',  # paragraph break (best)
        '\n',  # line break
        '。',  # Chinese period
        '！',  # Chinese exclamation
        '？',  # Chinese question mark
        '；',  # Chinese semicolon
        '，',  # Chinese comma
        ' ',  # space
        '',  # character by character (last resort)
    ]

    return _split_recursive(text, separators, chunk_size, chunk_overlap)


def _split_recursive(text, separators, chunk_size, chunk_overlap):
    """internal recursive splitting function"""

    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    # find the best separator that exists in the text
    best_sep = ''
    for sep in separators:
        if sep == '':
            best_sep = sep
            break
        if sep in text:
            best_sep = sep
            break

    # split by the chosen separator
    if best_sep:
        parts = text.split(best_sep)
    else:
        parts = list(text)  # character by character

    # merge parts into chunks of appropriate size
    chunks = []
    current_chunk = ''

    for part in parts:
        # if adding this part would exceed chunk_size
        test_chunk = current_chunk + best_sep + part if current_chunk else part

        if len(test_chunk) <= chunk_size:
            current_chunk = test_chunk
        else:
            # save current chunk
            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            # if this single part is bigger than chunk_size,
            # recursively split it with a finer separator
            if len(part) > chunk_size:
                remaining_seps = separators[separators.index(best_sep) + 1:] if best_sep in separators else ['']
                if not remaining_seps:
                    remaining_seps = ['']
                sub_chunks = _split_recursive(part, remaining_seps, chunk_size, chunk_overlap)
                chunks.extend(sub_chunks)
                current_chunk = ''
            else:
                current_chunk = part

    # don't forget the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # add overlap between chunks
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            # take the last chunk_overlap characters from previous chunk
            prev = chunks[i - 1]
            overlap_text = prev[-chunk_overlap:] if len(prev) > chunk_overlap else prev

            # find a clean break point in the overlap
            # (don't start mid-sentence)
            for char in ['。', '！', '？', '；', '\n', '，']:
                pos = overlap_text.find(char)
                if pos != -1:
                    overlap_text = overlap_text[pos + 1:]
                    break

            new_chunk = overlap_text.strip() + '\n' + chunks[i]
            overlapped.append(new_chunk.strip())

        chunks = overlapped

    return chunks


def main():
    # ===== Check input file =====
    if not os.path.exists(INPUT_FILE):
        print("ERROR: Cannot find input file: %s" % INPUT_FILE)
        print("Make sure you have run scraper_final.py first!")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ===== Step 1: Read file =====
    print("=" * 60)
    print("  RAG Chunker")
    print("=" * 60)

    print("\n[1/4] Reading file...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    print(f"  Raw text length: {len(raw_text):,} characters")

    # ===== Step 2: Clean =====
    print("\n[2/4] Cleaning text...")
    clean = clean_text(raw_text)
    print(f"  Cleaned text length: {len(clean):,} characters")
    print(f"  Removed: {len(raw_text) - len(clean):,} characters of noise")

    # ===== Step 3: Chunk =====
    print("\n[3/4] Splitting into chunks...")
    print(f"  Chunk size: {CHUNK_SIZE} chars")
    print(f"  Overlap: {CHUNK_OVERLAP} chars")

    chunks = recursive_split(clean, CHUNK_SIZE, CHUNK_OVERLAP)

    # remove empty or too-short chunks
    chunks = [c for c in chunks if len(c.strip()) > 50]

    print(f"  Generated {len(chunks)} chunks")

    # calculate stats
    lengths = [len(c) for c in chunks]
    avg_len = sum(lengths) // len(lengths) if lengths else 0
    min_len = min(lengths) if lengths else 0
    max_len = max(lengths) if lengths else 0

    print(f"  Average chunk size: {avg_len} chars")
    print(f"  Smallest chunk: {min_len} chars")
    print(f"  Largest chunk: {max_len} chars")

    # ===== Step 4: Save =====
    print("\n[4/4] Saving results...")

    # --- Save as JSON (for your RAG program to read) ---
    json_path = os.path.join(OUTPUT_DIR, "chunks.json")
    chunks_data = []
    for i, chunk in enumerate(chunks):
        chunks_data.append({
            "id": i,
            "content": chunk,
            "source": "Capital",
            "char_count": len(chunk),
        })

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=2)
    print(f"  JSON saved: {os.path.abspath(json_path)}")

    # --- Save as JSONL (one JSON per line, some tools prefer this) ---
    jsonl_path = os.path.join(OUTPUT_DIR, "chunks.jsonl")
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in chunks_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"  JSONL saved: {os.path.abspath(jsonl_path)}")

    # --- Save a preview file (for you to check) ---
    preview_path = os.path.join(OUTPUT_DIR, "chunks_preview.txt")
    with open(preview_path, 'w', encoding='utf-8') as f:
        f.write("RAG Chunks Preview\n")
        f.write(f"Total chunks: {len(chunks)}\n")
        f.write(f"Chunk size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}\n")
        f.write("=" * 60 + "\n\n")

        # show first 30 chunks
        show_count = min(30, len(chunks))
        for i in range(show_count):
            f.write(f"\n--- Chunk {i + 1} ({len(chunks[i])} chars) ---\n")
            f.write(chunks[i])
            f.write("\n")

        if len(chunks) > show_count:
            f.write(f"\n\n... and {len(chunks) - show_count} more chunks\n")

    print(f"  Preview saved: {os.path.abspath(preview_path)}")

    # ===== Summary =====
    print("\n" + "=" * 60)
    print("  DONE!")
    print("=" * 60)
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Average size: {avg_len} chars per chunk")
    print()
    print("  Output files:")
    print(f"    chunks.json     <- your RAG program reads this")
    print(f"    chunks.jsonl    <- alternative format")
    print(f"    chunks_preview.txt <- open this to check quality")
    print()
    print("  Next steps:")
    print("    1. Open chunks_preview.txt, check if chunks look reasonable")
    print("    2. Load chunks.json into your vector database")
    print("    3. Use an embedding model to vectorize each chunk")
    print("    4. Query with questions and retrieve relevant chunks")

    # show a few example chunks
    print("\n" + "=" * 60)
    print("  SAMPLE CHUNKS (first 3)")
    print("=" * 60)
    for i in range(min(3, len(chunks))):
        print(f"\n--- Chunk {i + 1} ({len(chunks[i])} chars) ---")
        # show first 200 chars of each
        preview = chunks[i][:200]
        if len(chunks[i]) > 200:
            preview += "..."
        print(preview)


if __name__ == "__main__":
    main()