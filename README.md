# Medical Chatbot using finetuned LLM and RAG on Pubmed dataset

This repository contains a medical chatbot using a finetuned LLM and RAG system on a Pubmed dataset.

See the [Documentation](./DOCUMENTATION.md) for more information.

## Installation and Running

### Requirements

- install pip requirements using `pip3 install -r requirements.txt`

- install nodejs (version 20.x) using e.g. https://github.com/nvm-sh/nvm

- install ollama: https://github.com/ollama/ollama

- install docker: https://docs.docker.com/engine/install/

### Setup

- pull the used llm using `ollama pull cniongolo/biomistral`

- pull chat-ui dependencies using `cd chat-ui-rag && npm i`

- pull the [dataset](https://www.dropbox.com/scl/fi/hyrmwrvcjqc5huue84ici/pubmed_data.zip?rlkey=hrbuq31wvsou9r8elsfmrxstq&dl=0)

### Starting

- start pinecone vectorstore using `python3 chat-ui-rag/src/lib/server/rag/pinecone/pineconeEndpoint.py`

- as alternative to pinecone: start OpenSearch using `docker run -d -p 9200:9200 -p 9600:9600 -e "discovery.type=single-node" -e "OPENSEARCH_INITIAL_ADMIN_PASSWORD=qaOllama2" opensearchproject/opensearch:latest`. Change "vectorStoreType" to "opensearch" at [.env](./chat-ui-rag/.env)

- start mongodb server using `docker run -d -p 27017:27017 --name mongo-chatui mongo:latest`

- start chat-ui dev server using `cd chat-ui-rag && npm run dev`

### Access

- access chat-ui with browser at http://localhost:5173

## Repository Overview

The main components of the repository are:

### User Interface: [ChatUI](./chat-ui-rag/)[^1]

Here you can find the user interface for the chatbot based on the open-source project. We expanded the project to include a RAG system inserting the content of scientific papers retrieved from the Pubmed dataset.

### RAG System: [RAG](./chat-ui-rag/src/lib/server/rag/)

Here you can find the RAG system used for the chatbot. It provides an endpoint for the chat-ui to query the RAG system for papers relevant to questions posed by the user. The retrieved papers are inserted into the user prompt at [buildPrompt.ts](./chat-ui-rag/src/lib/buildPrompt.ts).

### Data Preprocessing: [DataPreprocessing](./preprocessing.ipynb)

This notebook containes the code we used to retrieve and process the Pubmed dataset as well as upload embeddings of the papers to the Pinecone vectorstore.

### System Evaluation: [Evaluation](./evaluation/)

This folder contains the code and results of the evaluation of the chatbot system.

### Meetings: [Meetings](./meetings)

This folder contains the notes of the meetings we had during the project.

### Notes: [Notes](./notes)

This folder contains the notes we took during the project.

### Opensearch: [Opensearch](./opensearch)

<!-- DESCRIBE -->

[^1]: https://github.com/huggingface/chat-ui
