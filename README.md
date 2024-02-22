# chatbot-llm

- **Goal:** Create a chatbot that integrates with a searchable document database and uses OpenAI language models to retrieve information and answer user questions.
- **Requirements:**
  - Handle PDF uploads and process the content.
  - Efficiently search for relevant documents based on user questions.
  - Leverage a powerful language model to understand queries and generate informative answers.

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

---

1. **Technology Selection:**

   - **OpenAI API:** For access to GPT-3.5-turbo and text embedding features.
   - **Flask:** A lightweight web framework for creating the API endpoints.
   - **Chroma:** Database and embedding functionalities for vector-based search.
   - **PyPDF2:** Tool for loading and processing PDF files.

2. **Design and Architecture:**

   - **Web API Endpoints:**
     - `/upload-pdf` (Handles file uploads and document processing.)
     - `/similar_docs` (Finds relevant documents based on a question.)
     - `/answer_query` (Generates a comprehensive answer.)
   - **Document Handling:** Uploaded PDFs are split into chunks and embedded for storage in the Chroma database.
   - **Conversational AI Component:** Employs OpenAI's language model with a Retriever and Memory for context.

3. **Implementation:**

   - **Code Structure:** Modular design with clear separation between document handling, the conversational model, and API endpoints.
   - **Error Handling:** Implementation of try/except blocks and specific error responses for different failure scenarios.

4. **Testing:**

   - **Unit Tests:** Focused on isolated functions like file validation and document processing.
   - **Integration Tests:** Ensured end-to-end functionality, focusing on the full pipeline from PDF upload to answer generation.
   - **Manual Testing:** Used Postman (as a basic example) to experiment with API endpoints using various inputs.

5. **Challenges and Solutions**

- **Challenge 1: Efficient Document Search:** Indexing and searching large text collections can be computationally expensive.
  - **Solution:** Employed Chroma, specifically designed for vector-based search to address this performance challenge.
- **Challenge 2: Handling Complex Questions:** User queries can be ambiguous or require understanding broader context from documents.
  - **Solution:** Used OpenAI's powerful GPT-3.5-turbo model and integrated a memory component to retain conversation history.

---

**Self-Evaluation of Performance**

- **Successes:**
  - The chatbot can successfully process PDF files and store their contents in a searchable format.
  - Retrieval of relevant documents based on user queries works well in most cases.
  - Answers to questions are generally coherent and informative.
- **Areas for Improvement:**
  - Fine-tuning the language model with more domain-specific data could further improve answer quality.
  - More sophisticated error handling and logging for better debugging and monitoring.
  - Performance optimization to handle very large document collections.
