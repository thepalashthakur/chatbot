from flask import Flask, request, jsonify,render_template
from app.main import bp
from weaviate import Client, schema
import fitz  # PyMuPDF
from transformers import AutoTokenizer, AutoModel
from langchain.llms import OpenAI

import datetime
import getpass
import os

import openai
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings  # Assuming this is the correct one if they are identical
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Weaviate


API_KEY = ""
client = Client("http://localhost:8080")

pdf_path = "data_src.pdf"
title = "data_src"  # You might want to extract this or define it manually
prompt_template = "Given the context: {context}, what would be the best way to {question}?"
openai_llm = OpenAI(api_key=API_KEY)
context = "designing a VectorStore from a PDF in Weaviate"
question = "integrate this VectorStore with an AI model for semantic search"
prompt = prompt_template.format(context=context, question=question)
response = openai_llm(prompt)
current_date = datetime.datetime.now().date()
if current_date < datetime.date(2023, 9, 2):
    llm_name = "gpt-3.5-turbo-0301"
else:
    llm_name = "gpt-3.5-turbo"

openai.api_key  = API_KEY
os.environ["OPENAI_API_KEY"] = API_KEY
embedding = OpenAIEmbeddings()
persist_directory = 'docs/chroma/'
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

def generate_embeddings(text):
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return embeddings


def insert_document(title, content, vector):
    document = {
        "title": title,
        "content": content
    }
    # Insert the document with the vector
    client.data_object.create(document, class_name="Document", vector=vector)

@bp.route('/')
def index():
    return 'This is The Main Blueprint'


@bp.route('/query', methods=['POST'])
def handle_query():
    try:
        data = request.json
        question = data.get('question')
        query_embedding = embedding.embed_query(question)
        k = 5 # Number of similar items to retrieve
        vector_query = {
            "vector": query_embedding
        }
        results = client.query.get(
            "Document", 
            ["content"]  # Properties you want to retrieve
        ).with_near_vector(
            vector_query  # Use the correctly formatted query
        ).with_limit(k).do()
        try:    
            similar_result_queries = ""
            for result in results["data"]["Get"]["Document"]:
                similar_result_queries += result["content"]+" "
        except:
            pass

        prompt_template = "Given the context: {context}, what would be the best way to {question}?"
        prompt = prompt_template.format(context=similar_result_queries, question=question)
        response = openai_llm(prompt)
        vector = generate_embeddings(response)[0]
        insert_document("OpenAI Response", response, vector)  # Store in Weaviate        
        return jsonify({"response": response,"Success":True,"similar_result_queries":similar_result_queries})

    except Exception as e:
        return jsonify({"response": e,"Success":False})

@bp.route('/retrievalquery', methods=['POST'])
def askQuestion():
    try:
      data = request.json
      question = data.get('question')
      docs = vectordb.similarity_search(question,k=3)
      len(docs)
      llm = ChatOpenAI(model_name=llm_name, temperature=0)
      # Build prompt
      template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
      {context}
      Question: {question}
      Helpful Answer:"""
      QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)
      # Run chain
      qa_chain = RetrievalQA.from_chain_type(llm,
                                          retriever=vectordb.as_retriever(),
                                          return_source_documents=True,
                                          chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})


      result = qa_chain({"query": question})
      result["result"]
      memory = ConversationBufferMemory(
          memory_key="chat_history",
          return_messages=True
      )
      retriever=vectordb.as_retriever()
      qa = ConversationalRetrievalChain.from_llm(
          llm,
          retriever=retriever,
          memory=memory
      )
      result = qa({"question": question})
      return jsonify({"response": result['answer'],"Success":True})
    except Exception as e:
      return jsonify({"response": e,"Success":False})
        
      