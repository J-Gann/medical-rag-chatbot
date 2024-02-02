FROM ollama/ollama
WORKDIR /app
COPY . .

RUN apt update

RUN apt install python3 python3-pip -y

RUN pip install -U torch numpy sentencepiece transformers bitsandbytes accelerate Biopython pinecone-client fastapi uvicorn

CMD ["uvicorn", "rag_system:app", "--port 8080"]

EXPOSE 8080