# chatbot-llm

A chatbot using Flask, Vector DB, and OpenAI's GPT for semantic document handling and retrieval through direct and conversational queries.

Overview
This Python code, built using the Flask framework, demonstrates an application of LangChain and other AI libraries for the following core functions:
Document Processing: Extracting text content from uploaded PDF files, preparing textual data for storage within a vector database.
Similarity-Based Search: Identifying similar sections of text within stored documents based on a user-provided query.
Question Answering: Leveraging vector embeddings, large language models (LLMs), and conversational memory to answer questions about the content of the processed documents.

Running the Application:

API Key Setup
You need an API Key from OpenAI (https://beta.openai.com/account/api-keys) to use their services.
This section stores your API key in multiple ways to allow for flexibility in how it's used later in the code. 2. Importing Libraries
You can assign the API_KEY in this variable
API_KEY = "YOUR_API_KEY"

After Cloning the Git Repo run the following commands in the terminal.
cd CHATBOT-LLM
pip install -r requirements.txt
export FLASK_APP=app
export FLASK_DEBUG=True
flask run

API ROUTES

1. /upload-pdf:
   Allows users to upload a PDF file.
   Processes the PDF, stores relevant information in the database.

2. /similar_docs:
   Accepts a question as input.
   Searches the database for documents similar to the question.

3. /answer_query:
   Accepts a question as input.
   Uses the language model and document database to generate a comprehensive answer.

GENERATING RESPONSE

First you need to create embedding for the document you want to test on.
For which you can use this curl request.
This inclued a POST request to /upload-pdf route to create required embedding s and storing the data in vector database.
The post request expects form-data with file type containing pdf key and the file needed to be embedded.
curl --location 'http://localhost:5000/upload-pdf' \
--form 'pdf=@"/pathtodocument"'

Once the embeddings are genereated we can get the response by asking questions about the PDF
This will be a post request to answer_query route with json body containing "question"

curl --location 'http://127.0.0.1:5000/answer_query' \
--header 'Content-Type: application/json' \
--data '{
"question":"What is Encoder and Decoder Stacks"
}'

POST request to this provided the desired response.

---

ADDITIONAL DETAILS ABOUT PACKAGES USED

openai: The main library for interacting with OpenAI's models.
os: Lets you interact with the operating system (for setting environment variables).
chat_openai: Provides tools for building conversational AI applications.
chroma_embeddings: Enables storing and searching information in an organized way (vector database). 3. Initializing Components

openai_llm: An object for interacting with OpenAI language models (redundant, the second line preferred).
memory: Stores conversation history so the AI can maintain context.
llm_name: Specifies the powerful "gpt-3.5-turbo" language model.
llm: An object for having conversations with the OpenAI model. 4. Document Storage & Embedding

persist_directory: Where to store data related to the document collection.
embedding: A method to turn text into numerical representations (vectors) for efficient searching.
vectordb: A database designed to store and search these text embeddings. 5. Information Retrieval

retriever: A component that can search the document database for relevant information.
qa: The main conversational AI object, combining the language model, retriever, and memory for a comprehensive question-answering system.
