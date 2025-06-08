import os
from flask import Flask, request, jsonify
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from utils import clean_filename

load_dotenv()

app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/processar', methods=['POST'])
def processar():
    file = request.files['file']
    prompt = request.form.get('prompt')
    client_id = request.form.get('client_id')

    if not file:
        return jsonify({"erro": "Arquivo não enviado"}), 400

    filename = clean_filename(file.filename, client_id)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    if file.filename.endswith('.pdf'):
        loader = PyPDFLoader(filepath)
    elif file.filename.endswith('.csv'):
        loader = CSVLoader(filepath)
    else:
        return jsonify({"erro": "Formato de arquivo não suportado"}), 400

    documents = loader.load()
    embedding_function = HuggingFaceEmbeddings()  # usa emb. da OpenAI por enquanto
    db = Chroma.from_documents(documents, embedding_function, persist_directory=f"./db/{client_id}")
    db.persist()

    retriever = db.as_retriever()
    llm = ChatGroq(temperature=0, model_name="gemma2-9b-it")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    resposta = qa_chain.run(prompt)
    return jsonify({"resposta": resposta})

@app.route('/perguntar', methods=['POST'])
def perguntar():
    data = request.get_json()
    question = data.get('question')
    client_id = data.get('client_id')

    if not question or not client_id:
        return jsonify({"erro": "Parâmetros 'question' e 'client_id' são obrigatórios."}), 400

    db_path = f"./db/{client_id}"
    if not os.path.exists(db_path):
        return jsonify({"erro": "Base de conhecimento não encontrada para esse cliente."}), 404

    embedding_function = HuggingFaceEmbeddings()
    db = Chroma(persist_directory=db_path, embedding_function=embedding_function)
    retriever = db.as_retriever()

    llm = ChatGroq(temperature=0.7, model_name="llama-3.3-70b-versatile")  # corrigido o model_name
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # Prompt base usado com personalização
    prompt_template = """Você é um atendente virtual especializado em cursos online.
    Use as informações fornecidas para responder à pergunta abaixo de forma clara, simples e com tom amigável.

    Seja direto, prestativo e, se não souber a resposta com base nos dados, diga que a informação não está disponível.

    Agora responda à seguinte pergunta:
    {prompt}
    """

    final_prompt = prompt_template.replace("{prompt}", question)
    resposta = qa_chain.run(final_prompt)

    return jsonify({"resposta": resposta})

if __name__ == "__main__":
    app.run(debug=True)