# QA-INLPT-WS2023

1. Start the document-retrieval server (requires data file located at: https://www.dropbox.com/scl/fi/hyrmwrvcjqc5huue84ici/pubmed_data.zip?rlkey=hrbuq31wvsou9r8elsfmrxstq&dl=0): 

``python3 chat-ui-rag/src/lib/server/rag/pinecone/pineconeEndpoint.py``

2. Install and run ollama (see: https://github.com/ollama/ollama)

3. Download llama2 model

``ollama pull llama2``

4. Install nodejs (version 20.x)

``curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -``
``sudo apt-get install -y nodejs``

5. Install chat-ui dependencies

``cd chat-ui-rag && npm i``

6. Start mongodb server

``docker run -d -p 27017:27017 --name mongo-chatui mongo:latest``

7. Start the chat-ui dev server

``npm run dev``

8. Access chat-ui with browser at **port 5173**