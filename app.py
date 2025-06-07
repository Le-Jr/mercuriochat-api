import os
from flask import Flask, request, jsonify
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatGroq
from dotenv import load_dotenv
import os
from utils import clean_filename

load_dotenv()

app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/processar', methods=['POST'])
def processar():
    file = request.files['arquivo']
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
    embedding_function = OpenAIEmbeddings()  # usa emb. da OpenAI por enquanto
    db = Chroma.from_documents(documents, embedding_function, persist_directory=f"./db/{client_id}")
    db.persist()

    retriever = db.as_retriever()
    llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    resposta = qa_chain.run(prompt)
    return jsonify({"resposta": resposta})
