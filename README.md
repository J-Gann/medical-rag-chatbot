# QA-INLPT-WS2023

1. Start the document-retrieval server (requires data file located at: https://www.dropbox.com/scl/fi/hyrmwrvcjqc5huue84ici/pubmed_data.zip): 

``python3 chat-ui-rag/src/lib/server/rag/pinecone/pineconeEndpoint.py``

2. Install and run ollama (see: https://github.com/ollama/ollama)

3. Download llama2 model

``ollama pull llama2``

4. Install chat-ui dependencies

``cd chat-ui && npm i``

5. Start mongodb server

``docker run -d -p 27017:27017 --name mongo-chatui mongo:latest``

6. create .ev.local file with the following line

``MONGODB_URL=mongodb://127.0.0.1:27017``

6. Start the chat-ui dev server

``npm run dev``

7. Access chat-ui with browser at **port 5173**



