# QA-INLPT-WS2023

1. Start the document-retrieval server: 

``python3 chat-ui-rag/src/lib/server/rag/pinecone/pineconeEndpoint.py``

2. Install chat-ui dependencies

``cd chat-ui && npm i``

3. Start mongodb server

``docker run -d -p 27017:27017 --name mongo-chatui mongo:latest``

4. Start the chat-ui dev server

``npm run dev``

5. Access chat-ui with browser at **port 5173**



