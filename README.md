# API RAG - Sistema de Recuperação e Geração Aumentada

Uma API Flask robusta para processamento de documentos e consultas inteligentes usando RAG (Retrieval-Augmented Generation) com integração Groq e embeddings HuggingFace.

## 🚀 Funcionalidades

- **Processamento Assíncrono**: Upload e processamento de documentos PDF, CSV e TXT
- **Knowledge Base por Cliente**: Cada cliente tem sua própria base de conhecimento
- **Cache Inteligente**: Redis para cache de respostas frequentes
- **Rate Limiting**: Proteção contra abuso com limites configuráveis
- **Segurança**: Autenticação via API Key e validações robustas
- **Monitoramento**: Logs detalhados e health checks
- **Escalabilidade**: Thread pool para processamento paralelo

## 📋 Pré-requisitos

- Python 3.8+
- Redis Server
- Groq API Key

## 🛠️ Instalação

### 1. Clone o repositório

```bash
git clone <url-do-repositorio>
cd rag-api
```

### 2. Instale as dependências

```bash
pip install -r requirements.txt
```

### 3. Configure as variáveis de ambiente

Crie um arquivo `.env` na raiz do projeto:

```env
# Groq Configuration
GROQ_API_KEY=sua_groq_api_key_aqui

# Security
API_KEY=sua_api_key_secreta

# Directories
UPLOAD_FOLDER=./uploads
DB_FOLDER=./vector_db
SQLITE_DB=./rag_metadata.db

# Redis
REDIS_URL=redis://localhost:6379/0
CACHE_TTL=3600

# Rate Limiting
RATE_LIMIT_DEFAULT=100/hour

# Performance
MAX_WORKERS=4
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# CORS
ALLOWED_ORIGINS=http://localhost:3000,https://seu-frontend.com

# Environment
ENV_TYPE=production
```

### 4. Inicie o Redis

```bash
# Ubuntu/Debian
sudo systemctl start redis-server

# Docker
docker run -d -p 6379:6379 redis:alpine

# macOS com Homebrew
brew services start redis
```

### 5. Execute a aplicação

**Desenvolvimento:**

```bash
ENV_TYPE=development python app.py
```

**Produção:**

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## 📡 Endpoints da API

### Health Check

```http
GET /health
```

**Resposta:**

```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00",
  "redis": "connected"
}
```

### Processar Documento

```http
POST /processar
Content-Type: multipart/form-data
X-API-Key: sua_api_key
```

**Parâmetros:**

- `file`: Arquivo (PDF, CSV, TXT) - máx 50MB
- `client_id`: Identificador único do cliente

**Resposta:**

```json
{
  "job_id": "cliente123_a1b2c3d4e5f6g7h8",
  "status": "aceito",
  "mensagem": "Arquivo sendo processado. Use /status/{job_id} para acompanhar."
}
```

### Verificar Status do Processamento

```http
GET /status/{job_id}
X-API-Key: sua_api_key
```

**Resposta:**

```json
{
  "job_id": "cliente123_a1b2c3d4e5f6g7h8",
  "status": "completed",
  "created_at": "2024-01-15T10:00:00",
  "completed_at": "2024-01-15T10:02:30",
  "document_count": 45,
  "error_message": null
}
```

### Fazer Pergunta

```http
POST /perguntar
Content-Type: application/json
X-API-Key: sua_api_key
```

**Body:**

```json
{
  "question": "Qual é o procedimento para X?",
  "client_id": "cliente123"
}
```

**Resposta:**

```json
{
  "resposta": "Baseado nos documentos fornecidos, o procedimento para X é...",
  "cached": false,
  "response_time": 2.34
}
```

### Informações do Cliente

```http
GET /clientes/{client_id}/info
X-API-Key: sua_api_key
```

**Resposta:**

```json
{
  "client_id": "cliente123",
  "knowledge_base": {
    "created_at": "2024-01-15T09:00:00",
    "updated_at": "2024-01-15T10:02:30",
    "document_count": 3,
    "total_chunks": 150
  },
  "recent_jobs": [...]
}
```

## 🔧 Configuração Detalhada

### Rate Limiting

Por padrão, os limites são:

- `/processar`: 10 requests/minuto
- `/perguntar`: 50 requests/minuto
- `/status`: 30 requests/minuto
- Geral: 100 requests/hora

### Formatos Suportados

- **PDF**: Extração de texto completa
- **CSV**: Processamento de dados tabulares
- **TXT**: Arquivos de texto simples

### Cache

- Cache automático de respostas no Redis
- TTL configurável (padrão: 1 hora)
- Chave baseada em client_id + pergunta

## 🐳 Deploy com Docker

### Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

### docker-compose.yml

```yaml
version: "3.8"
services:
  rag-api:
    build: .
    ports:
      - "5000:5000"
    environment:
      - REDIS_URL=redis://redis:6379/0
      - GROQ_API_KEY=${GROQ_API_KEY}
      - API_KEY=${API_KEY}
    depends_on:
      - redis
    volumes:
      - ./uploads:/app/uploads
      - ./vector_db:/app/vector_db

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
```

## 📊 Monitoramento

### Logs

Os logs são salvos em `rag_api.log` e incluem:

- Processamento de documentos
- Performance de consultas
- Erros e exceções
- Status de conexões

### Métricas Importantes

- Tempo de processamento de documentos
- Cache hit rate
- Tempo de resposta das consultas
- Taxa de erro por endpoint

## 🔒 Segurança

- **Autenticação**: API Key obrigatória (exceto /health)
- **Rate Limiting**: Proteção contra DDoS e abuso
- **Validação**: Sanitização de nomes de arquivos
- **CORS**: Controle de origens permitidas
- **Logs**: Auditoria completa de ações

## 🚨 Solução de Problemas

### Erro: "Redis disconnected"

```bash
# Verifique se o Redis está rodando
redis-cli ping

# Reinicie o serviço
sudo systemctl restart redis-server
```

### Erro: "GROQ_API_KEY não configurada"

Verifique se a variável está no `.env` e se a API key é válida.

### Erro: "Knowledge base não encontrada"

O cliente precisa primeiro fazer upload de documentos via `/processar`.

### Performance lenta

- Aumente `MAX_WORKERS` para mais threads
- Reduza `CHUNK_SIZE` se os documentos são pequenos
- Verifique se o Redis está funcionando para cache

## 📈 Escalabilidade

Para ambientes de alta demanda:

1. **Load Balancer**: Use nginx ou similar
2. **Múltiplas Instâncias**: Scale horizontal com gunicorn
3. **Redis Cluster**: Para cache distribuído
4. **Monitoring**: Implemente Prometheus + Grafana

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para detalhes.

## 📞 Suporte

Para dúvidas ou problemas:

- Abra uma issue no GitHub
- Consulte os logs em `rag_api.log`
- Verifique a documentação da API Groq

---

**Versão**: 1.0.0  
**Última atualização**: Janeiro 2024
