from flask import Flask, request, jsonify, render_template, redirect, url_for
from llama_cpp import Llama
import json
import faiss
import numpy as np
from typing import Tuple, Dict
import tiktoken
from markdown import markdown

# Настройка сервера Flask
app = Flask(__name__)

# Пути к моделям
model_path = "/path/to/your/llama_gguf/model"
emb_mod = "/path/to/your/embedding_gguf/model"
max_input_tokens = 512  # Максимальное количество токенов во вводе
n_ctx = 3072  # Размер памяти сети
n_gpu_layers = -1  # Кол-во слоев, выгружаемых на GPU(-1 все)
chat_format = "llama-3"  # Формат чата
max_res_tok = 250  # Количество токенов в ответе

# Настройка токенизатора
encoder = tiktoken.get_encoding("cl100k_base")

# Загрузка источников и индексация
index_path = "/path/to/your/faiss/index/file"
mapping_path = index_path.replace(".index", "_mapping.json")

with open(mapping_path, "r", encoding='utf-8') as f:
    source_mapping = json.load(f)

class model_llama():

    def __init__(self, lm_model_path, emb_model_path, source_mapping, index_path, n_ctx = 2042, n_gpu_layers = None, chat_format = "llama-3", max_res_tok = 250) -> None:
        self.llama = Llama(model_path=lm_model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, chat_format=chat_format)  # Загрузка модели Llama
        self.emb_llama = Llama(model_path=emb_model_path, n_gpu_layers=0, embedding=True)  # Загрузка эмбединг модели
        self.source_mapping = source_mapping
        self.max_res_tok = max_res_tok
        self.index = faiss.read_index(index_path)
        self.history = []  # для сохранения истории запросов

    # Функция для генерации эмбеддингов текста
    def embed_text(self, text: str) -> np.ndarray:
        result = self.emb_llama.embed(text)
        return np.array(result).astype(np.float32)

    # Поиск наиболее релевантного фрагмента
    def search_most_relevant_chunk(self, query: str) -> Tuple[int, str, str]:
        query_embedding = self.embed_text(query)
        distances, indices = self.index.search(query_embedding.reshape(1, -1), 1)
        idx = indices[0][0]
        mapping_info = self.source_mapping[str(idx)]
        source_id = mapping_info["source_id"]
        chunk_idx = mapping_info["chunk_idx"]
        chunk_text = source_mapping[str(chunk_idx + source_id)]['text_chunk']
        chunk_ist = source_mapping[str(chunk_idx + source_id)]['source_name']
        return int(source_id), chunk_text, chunk_ist

    # Эмуляция запроса к API
    def ask_internal(self, data: Dict[str, any]) -> Dict[str, any]:
        query = data.get('query', '')
        use_sources = data.get('use_sources', True)

        if use_sources:
            # Поиск самого релевантного фрагмента
            source_id, chunk, ist = self.search_most_relevant_chunk(query)

            # Формирование контекста для GPT
            context = f"Источник : {chunk}\n\nОтветь на следующий вопрос: {query}"
            prompt = context
        else:
            prompt = f"Ответь на следующий вопрос: {query}"

        try:
            # Генерация ответа
            response = self.llama(prompt=prompt, max_tokens=self.max_res_tok) 

            result = {
                "query": query,
                "response": response["choices"][0]["text"],
                "sources": [{"index": 1, "text": ist, "source_id": source_id}] if use_sources else []
            }
        except Exception as e:
            result = {
                "query": query,
                "response": f"Ошибка: {str(e)}. Попробуйте обновить GPT.",
                "sources": []
            }

        # Добавление ответа в историю
        self.history.append({
            "query": query,
            "response": result["response"],
            "sources": result["sources"]
        })

        return result

# Функция для подсчета токенов
def count_tokens(text: str) -> int:
    return len(encoder.encode(text))

# Загружаем модель
llama = model_llama(model_path, emb_mod, source_mapping, index_path, n_ctx, n_gpu_layers, chat_format, max_res_tok)

# Обновление модели 
@app.route('/update_gpt', methods=['POST'])
def update_gpt():
    global llama
    llama = model_llama(model_path, emb_mod, source_mapping, n_ctx, n_gpu_layers, chat_format, max_res_tok)
    return redirect(url_for('chat'))

# Графический интерфейс
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST', 'GET'])
def chat():
    global llama
    query = request.form.get('query', '')
    use_sources = request.form.get('use_sources') == "on"

    if request.method == 'POST' and query:
        response = llama.ask_internal({"query": query, "use_sources": use_sources})

    # Формирование истории в Markdown
    conversation_md = ""
    for entry in llama.history:
        sources_md = "\n\n".join([f"### Источник {src['text']}" for src in entry["sources"]])
        full_md = f"{sources_md}\n\n### Ответ:\n{entry['response']}" if entry["sources"] else f"### Ответ:\n{entry['response']}"
        conversation_md += f"#### Вопрос: {entry['query']}\n{full_md}\n\n"

    conversation_md = markdown(conversation_md)

    return render_template('chat.html', conversation=conversation_md, query=query, use_sources=use_sources)

# Подсчет токенов во вводе
@app.route('/count_tokens', methods=['POST'])
def count_tokens_endpoint():
    data = request.get_json()
    query = data.get('query', '')
    token_count = count_tokens(query)
    is_valid = token_count <= max_input_tokens
    return jsonify({"count": f"{token_count}/{max_input_tokens} токенов", "valid": is_valid})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)