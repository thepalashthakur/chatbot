# chatbot

The project developed a chatbot using Flask, Weaviate, and OpenAI's GPT for semantic document handling and retrieval through direct and conversational queries.

<!-- Request to use Weaviate approach   -->

curl --location 'http://localhost:5000/query' \
--header 'Content-Type: application/json' \
--data '{
"question":" what is Encoder and Decoder Stack"
}'

<!-- Request to use Chroma approach -->

curl --location 'http://localhost:5000/retrievalquery' \
--header 'Content-Type: application/json' \
--data '{
"question":" what is Encoder and Decoder Stack"
}'
