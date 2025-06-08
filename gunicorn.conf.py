# ================================
# gunicorn.conf.py
# ================================
import os

# Server socket
bind = "127.0.0.1:5001"  # Porta diferente da sua app principal
backlog = 2048

# Worker processes
workers = 4
worker_class = "sync"
worker_connections = 1000
timeout = 120
keepalive = 2

# Restart workers after this many requests
max_requests = 1000
max_requests_jitter = 50

# Logging
accesslog = "/var/log/rag-api/access.log"
errorlog = "/var/log/rag-api/error.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = 'rag-api'

# Server mechanics
daemon = False
pidfile = '/var/run/rag-api.pid'
user = 'www-data'  # ou seu usuário
group = 'www-data'

# SSL (se necessário)
# keyfile = "/path/to/keyfile"
# certfile = "/path/to/certfile"
