import os
import logging
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import redis
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

from flask import Flask, request, jsonify, g
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_chroma.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

import sqlite3
from contextlib import contextmanager
import json
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

load_dotenv()

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configurações
class Config:
    # Segurança
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS = {'pdf', 'csv', 'txt'}
    
    # Diretórios
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', './uploads')
    DB_FOLDER = os.getenv('DB_FOLDER', './vector_db')
    SQLITE_DB = os.getenv('SQLITE_DB', './rag_metadata.db')
    
    # Redis
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    CACHE_TTL = int(os.getenv('CACHE_TTL', '3600'))  # 1 hora
    
    # Rate Limiting
    RATELIMIT_STORAGE_URL = REDIS_URL
    RATE_LIMIT_DEFAULT = os.getenv('RATE_LIMIT_DEFAULT', '100/hour')
    
    # Groq
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    
    # Performance
    MAX_WORKERS = int(os.getenv('MAX_WORKERS', '4'))
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '1000'))
    CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '200'))

# Status de processamento
class ProcessingStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class ProcessingJob:
    job_id: str
    client_id: str
    filename: str
    status: ProcessingStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

# Inicialização
app = Flask(__name__)
app.config.from_object(Config)

# Middleware de segurança
CORS(app, origins=os.getenv('ALLOWED_ORIGINS', '*').split(','))

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=[Config.RATE_LIMIT_DEFAULT],
    storage_uri=Config.RATELIMIT_STORAGE_URL
)

# Redis para cache
try:
    redis_client = redis.from_url(Config.REDIS_URL, decode_responses=True)
    redis_client.ping()
    logger.info("Redis conectado com sucesso")
except Exception as e:
    logger.error(f"Erro ao conectar Redis: {e}")
    redis_client = None

# Thread pool para processamento assíncrono
executor = ThreadPoolExecutor(max_workers=Config.MAX_WORKERS)

# Utilitários
def allowed_file(filename: str) -> bool:
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def generate_safe_filename(original_filename: str, client_id: str) -> str:
    """Gera nome de arquivo seguro com hash único"""
    timestamp = int(time.time())
    random_suffix = secrets.token_hex(8)
    ext = secure_filename(original_filename).rsplit('.', 1)[1].lower()
    return f"{client_id}_{timestamp}_{random_suffix}.{ext}"

def get_cache_key(client_id: str, question: str) -> str:
    """Gera chave de cache para pergunta"""
    content = f"{client_id}:{question}"
    return f"rag_cache:{hashlib.md5(content.encode()).hexdigest()}"

# Database utilities
@contextmanager
def get_db_connection():
    conn = sqlite3.connect(Config.SQLITE_DB)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_database():
    """Inicializa banco de dados SQLite para metadados"""
    with get_db_connection() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS processing_jobs (
                job_id TEXT PRIMARY KEY,
                client_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                original_filename TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                error_message TEXT,
                file_hash TEXT,
                document_count INTEGER
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS client_knowledge_base (
                client_id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                document_count INTEGER DEFAULT 0,
                total_chunks INTEGER DEFAULT 0
            )
        ''')
        conn.commit()

# Validações de segurança
def validate_client_id(client_id: str) -> bool:
    """Valida formato do client_id"""
    return client_id and client_id.replace('_', '').replace('-', '').isalnum() and len(client_id) <= 50

def calculate_file_hash(filepath: str) -> str:
    """Calcula hash SHA-256 do arquivo"""
    hash_sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

# Processamento assíncrono
def process_document_async(job_id: str, client_id: str, filepath: str, original_filename: str):
    """Processa documento de forma assíncrona"""
    try:
        logger.info(f"Iniciando processamento do job {job_id}")
        
        # Atualiza status para processando
        with get_db_connection() as conn:
            conn.execute(
                "UPDATE processing_jobs SET status = ? WHERE job_id = ?",
                (ProcessingStatus.PROCESSING.value, job_id)
            )
            conn.commit()
        
        # Carrega documento
        if original_filename.endswith('.pdf'):
            loader = PyPDFLoader(filepath)
        elif original_filename.endswith('.csv'):
            loader = CSVLoader(filepath)
        else:
            raise ValueError(f"Formato não suportado: {original_filename}")
        
        documents = loader.load()
        
        # Split em chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        splits = text_splitter.split_documents(documents)
        
        # Cria embeddings e vector store
        embedding_function = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        db_path = os.path.join(Config.DB_FOLDER, client_id)
        os.makedirs(db_path, exist_ok=True)
        
        # Verifica se já existe DB para este cliente
        if os.path.exists(os.path.join(db_path, "chroma.sqlite3")):
            # Carrega DB existente e adiciona novos documentos
            db = Chroma(persist_directory=db_path, embedding_function=embedding_function)
            db.add_documents(splits)
        else:
            # Cria novo DB
            db = Chroma.from_documents(
                splits, 
                embedding_function, 
                persist_directory=db_path
            )
        
        # db.persist()
        
        # Calcula hash do arquivo
        file_hash = calculate_file_hash(filepath)
        
        # Atualiza job como completo
        with get_db_connection() as conn:
            conn.execute(
                """UPDATE processing_jobs 
                   SET status = ?, completed_at = ?, document_count = ?, file_hash = ?
                   WHERE job_id = ?""",
                (ProcessingStatus.COMPLETED.value, datetime.now(), len(splits), file_hash, job_id)
            )
            
            # Atualiza ou cria registro do cliente
            conn.execute(
                """INSERT OR REPLACE INTO client_knowledge_base 
                   (client_id, updated_at, document_count, total_chunks)
                   VALUES (?, ?, 
                           COALESCE((SELECT document_count FROM client_knowledge_base WHERE client_id = ?), 0) + 1,
                           COALESCE((SELECT total_chunks FROM client_knowledge_base WHERE client_id = ?), 0) + ?)""",
                (client_id, datetime.now(), client_id, client_id, len(splits))
            )
            conn.commit()
        
        # Remove arquivo temporário
        try:
            os.remove(filepath)
        except OSError:
            pass
            
        logger.info(f"Job {job_id} processado com sucesso")
        
    except Exception as e:
        logger.error(f"Erro no processamento do job {job_id}: {str(e)}")
        
        # Atualiza job com erro
        with get_db_connection() as conn:
            conn.execute(
                "UPDATE processing_jobs SET status = ?, error_message = ? WHERE job_id = ?",
                (ProcessingStatus.ERROR.value, str(e), job_id)
            )
            conn.commit()

# Middleware de autenticação simples (baseado em API key)
@app.before_request
def authenticate():
    """Middleware de autenticação básica"""
    # Pula autenticação para health check
    if request.endpoint == 'health':
        return
    
    api_key = request.headers.get('X-API-Key')
    expected_key = os.getenv('API_KEY')
    
    if not expected_key:
        logger.warning("API_KEY não configurada - modo desenvolvimento")
        return
    
    if not api_key or api_key != expected_key:
        return jsonify({"erro": "API key inválida"}), 401

# Rotas
@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "redis": "connected" if redis_client else "disconnected"
    }
    return jsonify(status)

@app.route('/processar', methods=['POST'])
@limiter.limit("10/minute")
def processar():
    """Processa arquivo e cria knowledge base"""
    try:
        # Validações
        if 'file' not in request.files:
            return jsonify({"erro": "Arquivo não enviado"}), 400
        
        file = request.files['file']
        client_id = request.form.get('client_id')
        
        if not file or file.filename == '':
            return jsonify({"erro": "Arquivo inválido"}), 400
        
        if not client_id or not validate_client_id(client_id):
            return jsonify({"erro": "client_id inválido"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"erro": "Formato de arquivo não suportado"}), 400
        
        # Salva arquivo
        filename = generate_safe_filename(file.filename, client_id)
        filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        file.save(filepath)
        
        # Cria job de processamento
        job_id = f"{client_id}_{secrets.token_hex(16)}"
        
        with get_db_connection() as conn:
            conn.execute(
                """INSERT INTO processing_jobs 
                   (job_id, client_id, filename, original_filename, status)
                   VALUES (?, ?, ?, ?, ?)""",
                (job_id, client_id, filename, file.filename, ProcessingStatus.PENDING.value)
            )
            conn.commit()
        
        # Submete para processamento assíncrono
        executor.submit(process_document_async, job_id, client_id, filepath, file.filename)
        
        logger.info(f"Job {job_id} criado para cliente {client_id}")
        
        return jsonify({
            "job_id": job_id,
            "status": "aceito",
            "mensagem": "Arquivo sendo processado. Use /status/{job_id} para acompanhar."
        }), 202
        
    except RequestEntityTooLarge:
        return jsonify({"erro": "Arquivo muito grande"}), 413
    except Exception as e:
        logger.error(f"Erro em /processar: {str(e)}")
        return jsonify({"erro": "Erro interno do servidor"}), 500

@app.route('/status/<job_id>', methods=['GET'])
@limiter.limit("30/minute")
def check_status(job_id: str):
    """Verifica status do processamento"""
    try:
        with get_db_connection() as conn:
            row = conn.execute(
                "SELECT * FROM processing_jobs WHERE job_id = ?", 
                (job_id,)
            ).fetchone()
        
        if not row:
            return jsonify({"erro": "Job não encontrado"}), 404
        
        return jsonify({
            "job_id": row["job_id"],
            "status": row["status"],
            "created_at": row["created_at"],
            "completed_at": row["completed_at"],
            "error_message": row["error_message"],
            "document_count": row["document_count"]
        })
        
    except Exception as e:
        logger.error(f"Erro em /status: {str(e)}")
        return jsonify({"erro": "Erro interno do servidor"}), 500

@app.route('/perguntar', methods=['POST'])
@limiter.limit("50/minute")
def perguntar():
    """Responde pergunta baseada na knowledge base"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"erro": "JSON inválido"}), 400
        
        question = data.get('question', '').strip()
        client_id = data.get('client_id', '').strip()
        
        if not question or not client_id:
            return jsonify({"erro": "Parâmetros 'question' e 'client_id' são obrigatórios"}), 400
        
        if not validate_client_id(client_id):
            return jsonify({"erro": "client_id inválido"}), 400
        
        # Verifica cache
        cache_key = get_cache_key(client_id, question)
        if redis_client:
            cached_response = redis_client.get(cache_key)
            if cached_response:
                logger.info(f"Cache hit para cliente {client_id}")
                return jsonify({"resposta": cached_response, "cached": True})
        
        # Verifica se existe knowledge base
        db_path = os.path.join(Config.DB_FOLDER, client_id)
        if not os.path.exists(os.path.join(db_path, "chroma.sqlite3")):
            return jsonify({"erro": "Knowledge base não encontrada para este cliente"}), 404
        
        # Carrega knowledge base
        embedding_function = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        db = Chroma(persist_directory=db_path, embedding_function=embedding_function)
        retriever = db.as_retriever(search_kwargs={"k": 10})
        
        # Configura LLM
        if not Config.GROQ_API_KEY:
            return jsonify({"erro": "Configuração LLM não encontrada"}), 500
        
        llm = ChatGroq(
            temperature=0.7,
            model_name="llama3-70b-8192",
            groq_api_key=Config.GROQ_API_KEY
        )
        
    
        
        # Template de prompt
        prompt_template = PromptTemplate(
        input_variables=['context', 'question'],
        template= """Você é um assistente virtual especializado em responder perguntas com base nos documentos fornecidos.

Instruções:
- Use APENAS as informações dos documentos fornecidos
- Seja claro, direto e útil
- Se a informação não estiver disponível nos documentos, diga isso claramente
- Mantenha um tom profissional e amigável
- Não invente ou especule informações

contexto: {context}
Pergunta: {question}

Resposta:"""
)
          # Cria chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type = "stuff",
            retriever=retriever,
            return_source_documents=False,
            chain_type_kwargs = {"prompt": prompt_template},

            
        )
        
        
        # Gera resposta
        start_time = time.time()
        logger.info(f"Pergunta feita: {question}")
        result = qa_chain.invoke({"query": question})
        logger.info(f"Resultado do invoke: {result}")
        resposta = result['result']
        response_time = time.time() - start_time
        
        # Salva no cache
        if redis_client:
            redis_client.setex(cache_key, Config.CACHE_TTL, resposta)
        
        logger.info(f"Pergunta processada para cliente {client_id} em {response_time:.2f}s")
        
        return jsonify({
            "resposta": resposta,
            "cached": False,
            "response_time": round(response_time, 2)
        })
        
    except Exception as e:
        logger.error(f"Erro em /perguntar: {str(e)}")
        return jsonify({"erro": "Erro interno do servidor"}), 500

@app.route('/clientes/<client_id>/info', methods=['GET'])
@limiter.limit("20/minute")
def client_info(client_id: str):
    """Informações sobre knowledge base do cliente"""
    try:
        if not validate_client_id(client_id):
            return jsonify({"erro": "client_id inválido"}), 400
        
        with get_db_connection() as conn:
            # Info da knowledge base
            kb_info = conn.execute(
                "SELECT * FROM client_knowledge_base WHERE client_id = ?",
                (client_id,)
            ).fetchone()
            
            # Jobs do cliente
            jobs = conn.execute(
                """SELECT job_id, original_filename, status, created_at, completed_at 
                   FROM processing_jobs 
                   WHERE client_id = ? 
                   ORDER BY created_at DESC 
                   LIMIT 10""",
                (client_id,)
            ).fetchall()
        
        if not kb_info and not jobs:
            return jsonify({"erro": "Cliente não encontrado"}), 404
        
        return jsonify({
            "client_id": client_id,
            "knowledge_base": dict(kb_info) if kb_info else None,
            "recent_jobs": [dict(job) for job in jobs]
        })
        
    except Exception as e:
        logger.error(f"Erro em /clientes/{client_id}/info: {str(e)}")
        return jsonify({"erro": "Erro interno do servidor"}), 500

# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({"erro": "Arquivo muito grande"}), 413

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({"erro": "Muitas requisições. Tente novamente mais tarde."}), 429

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Erro interno: {str(e)}")
    return jsonify({"erro": "Erro interno do servidor"}), 500

# Inicialização
if __name__ == "__main__":
    # Cria diretórios necessários
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(Config.DB_FOLDER, exist_ok=True)
    
    # Inicializa banco de dados
    init_database()
    
    # Modo desenvolvimento apenas para testes locais
    if os.getenv('ENV_TYPE') == 'development':
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print(os.environ['ENV_TYPE'])
        print("Use um servidor WSGI como Gunicorn para produção!")
        print("Exemplo: gunicorn -w 4 -b 0.0.0.0:5000 app:app")