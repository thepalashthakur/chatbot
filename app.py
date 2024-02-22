from flask import Flask, request, jsonify
import fitz  # PyMuPDF
from werkzeug.utils import secure_filename
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
import os
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
import openai
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
import traceback

# 1. API Key Setup
API_KEY = ""
openai.api_key  = API_KEY
os.environ["OPENAI_API_KEY"] = API_KEY
openai_llm = OpenAI(api_key=API_KEY)
openai.api_key  = os.environ['OPENAI_API_KEY']
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
llm_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=llm_name, temperature=0)
persist_directory = 'docs/chroma/'
embedding = OpenAIEmbeddings()
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)
retriever=vectordb.as_retriever()
qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    memory=memory
)

app = Flask(__name__)
@app.route('/upload-pdf', methods=['POST'])
def upload_pdf():
    if 'pdf' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['pdf']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join("/tmp", filename)
        file.save(file_path)
        
        r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=0, 
        separators=["\n\n", "\n", " ", ""]
        )
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        docs = r_splitter.split_documents(pages)
        vectordb = Chroma.from_documents(
            documents=docs,
            embedding=embedding,
            persist_directory=persist_directory
        )
        os.remove(file_path)  # Clean up the stored file
        return jsonify({"text": docs[0].page_content}), 200
    else:
        return jsonify({"error": "Invalid file format"}), 400
    
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ['pdf']

@app.route('/similar_docs', methods=['POST'])
def get_similar_documents():
        try:
            data = request.json
            question = data.get('question')
            docs = vectordb.similarity_search(question,k=3)
            if len(docs)> 0:
                return jsonify({"DOCS_0": docs[0].page_content,"DOCS_1": docs[1].page_content,"DOCS_2": docs[2].page_content,"question":question}), 400
            else:
                return jsonify({"DOCS": "No similar content found","question":question}), 400
        except  Exception as e:
            return jsonify({"error": str(e),"question":question,"error_details":traceback.format_exc()}), 400
        
@app.route('/answer_query', methods=['POST'])
def answer_query():
    try:
        data = request.json
        question = data.get('question')
        result = qa({"question": question})
        return jsonify({"DOCS": result['answer'],"question":question}), 200
    except  Exception as e:
        return jsonify({"error": str(e),"question":question,"error_details":traceback.format_exc()}), 400
    
@app.errorhandler(404)
def page_not_found(e):
    # note that we set the 404 status explicitly
    return jsonify({"error": "Invalid Request, Page Not Found"}), 404

if __name__ == '__main__':
    app.run(debug=True)
