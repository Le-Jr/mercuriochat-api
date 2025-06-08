import streamlit as st
import requests
import time
import os

# Configurações da API
API_URL = "http://localhost:5000"
CLIENT_ID = "teste123"
API_KEY = None  # Defina sua API_KEY aqui se necessário, ex.: "sua-api-key"

# Função para processar arquivo
def processar_arquivo(file):
    url = f"{API_URL}/processar"
    files = {"file": (file.name, file, file.type)}
    data = {"client_id": CLIENT_ID}
    headers = {"X-API-Key": API_KEY} if API_KEY else {}
    
    try:
        response = requests.post(url, files=files, data=data, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": f"Erro ao processar arquivo: {str(e)}"}

# Função para verificar status
def verificar_status(job_id):
    url = f"{API_URL}/status/{job_id}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": f"Erro ao verificar status: {str(e)}"}

# Função para fazer pergunta
def fazer_pergunta(question):
    url = f"{API_URL}/perguntar"
    payload = {"question": question, "client_id": CLIENT_ID}
    headers = {"Content-Type": "application/json", "X-API-Key": API_KEY} if API_KEY else {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": f"Erro ao fazer pergunta: {str(e)}"}

# Interface Streamlit
st.title("Mercúrio API - Interface de RAG")

# Seção para processar arquivo
st.header("Processar Arquivo")
uploaded_file = st.file_uploader("Escolha um arquivo (PDF ou CSV)", type=["pdf", "csv"])
if uploaded_file:
    if st.button("Processar"):
        with st.spinner("Processando arquivo..."):
            result = processar_arquivo(uploaded_file)
            if "error" in result:
                st.error(result["error"])
            else:
                st.success(f"Arquivo aceito! Job ID: {result['job_id']}")
                st.session_state["job_id"] = result["job_id"]

# Seção para verificar status
if "job_id" in st.session_state:
    st.header("Verificar Status")
    if st.button("Checar Status"):
        with st.spinner("Verificando status..."):
            status = verificar_status(st.session_state["job_id"])
            if "error" in status:
                st.error(status["error"])
            else:
                st.info(f"Status: {status['status']}")
                if status["status"] == "completed":
                    st.success("Arquivo processado com sucesso! Pode perguntar agora.")

# Seção para fazer perguntas
st.header("Fazer Pergunta")
question = st.text_input("Digite sua pergunta:", placeholder="Ex.: Quantos cursos estão no documento?")
if st.button("Enviar Pergunta"):
    if question:
        with st.spinner("Buscando resposta..."):
            result = fazer_pergunta(question)
            if "error" in result:
                st.error(result["error"])
            else:
                st.write(f"**Resposta**: {result['resposta']}")
                st.write(f"**Tempo de resposta**: {result['response_time']}s")
                st.write(f"**Cacheado**: {'Sim' if result['cached'] else 'Não'}")
    else:
        st.warning("Por favor, digite uma pergunta!")

# Rodapé
st.markdown("---")
st.write("Feito com 💪 por Leandro, powered by Streamlit e Mercúrio API!")