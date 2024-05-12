import os
import json
import faiss
import numpy as np
from llama_cpp import Llama
from docx import Document
import mammoth
import fitz
import chardet
from typing import List

# Конфигурация
model_path = "/path/to/your/embedding_gguf/model"
directory_path = "/path/to/save/files"
index_path = "/path/to/your/faiss/index/file"
mapping_path = index_path.replace(".index", "_mapping.json")
dimension = 768  # Размер эмбеддинга
max_tokens_per_chunk = 256  # Ограничение на число токенов

# Загрузка модели
llama = Llama(model_path=model_path, embedding=True, n_gpu_layers=0)

# Функции для извлечения текста из различных файлов
def extract_text_from_txt(file_path: str) -> str:
    """Извлечение текста из txt файла с определением кодировки."""
    with open(file_path, "rb") as f:
        raw_data = f.read()
    result = chardet.detect(raw_data)
    encoding = result["encoding"]
    with open(file_path, "r", encoding=encoding) as f:
        return f.read().replace('\n', '')

def extract_text_from_pdf(file_path: str) -> str:
    """Извлечение текста из pdf файла."""
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text.replace('\n', '')

def extract_text_from_docx(file_path: str) -> str:
    """Извлечение текста из docx файла."""
    doc = Document(file_path)
    return " ".join([paragraph.text for paragraph in doc.paragraphs]).replace('\n', '')

def extract_text_from_doc(file_path: str) -> str:
    """Извлечение текста из doc файла."""
    with open(file_path, "rb") as doc_file:
        result = mammoth.extract_raw_text(doc_file)
    return result.value.replace('\n', '')

def extract_text_from_file(file_path: str) -> str:
    """Извлечение текста из любого поддерживаемого типа файла."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".txt":
        return extract_text_from_txt(file_path)
    elif ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    elif ext == ".doc":
        return extract_text_from_doc(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def extract_texts_from_directory(directory: str) -> dict:
    """Извлечение текста из всех файлов в директории и сохранение их в словарь."""
    texts = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                texts[file] = extract_text_from_file(file_path)
                print(f"Extracted text from {file_path}")
            except Exception as e:
                print(f"Error extracting text from {file_path}: {e}")
    return texts

def save_texts_to_json(texts: dict, output_path: str):
    """Сохранение текстов в JSON файл."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=4)

# Извлечение текста из всех файлов в директории и сохранение в JSON
texts = extract_texts_from_directory(directory_path)

# Функция для генерации эмбеддингов
def embed_text(text: str) -> np.ndarray:
    result = llama.embed(text)
    return np.array(result).astype(np.float32)

# Разбивает текст на фрагменты
def tokenize_and_split(text: str, max_tokens: int) -> List[str]:
    tokens = text.split()
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [" ".join(chunk) for chunk in chunks]

# Разбиение и индексирование всех текстов
chunks = []
source_mapping = {}

for idx, (source_name, source_text) in enumerate(texts.items()):
    text_chunks = tokenize_and_split(source_text, max_tokens_per_chunk)
    for chunk_idx, chunk in enumerate(text_chunks):
        chunks.append(chunk)
        source_mapping[len(chunks) - 1] = {"source_id": idx, "chunk_idx": chunk_idx, "source_name": source_name, "text_chunk": chunk}

# Создание эмбеддингов для всех фрагментов
embeddings = np.array([embed_text(chunk) for chunk in chunks], dtype=np.float32)

# Создание и сохранение FAISS индекса
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
faiss.write_index(index, index_path)

# Сохранение сопоставления индекса с исходными текстами
with open(mapping_path, "w", encoding="utf-8") as f:
    json.dump(source_mapping, f, ensure_ascii=False, indent=4)

print(f"FAISS index saved to {index_path}")