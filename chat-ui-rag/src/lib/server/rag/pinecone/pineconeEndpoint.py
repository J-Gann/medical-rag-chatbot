import torch
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from Bio import Medline
import ollama
from fastapi import FastAPI
import uvicorn

if torch.cuda.is_available():
  device = "cuda:0"
else:
  device = "cpu"
device

records = {}
missed = 0

with open("/home/paperspace/QA-INLPT-WS2023/chat-ui-rag/src/lib/server/rag/pinecone/pubmed_data") as stream:
    for article in Medline.parse(stream):

        if not "PMID" in article:
            missed += 1
            continue

        if not "TI" in article:
            missed += 1
            continue

        if not "FAU" in article:
            missed += 1
            continue

        if not "DP" in article:
            missed += 1
            continue

        if not "AB" in article:
            missed += 1
            continue
        
        records[article["PMID"]] = article

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

index_name = 'pubmed'
pinecone = Pinecone(api_key="7377f10c-e728-4408-b2a6-3ac28daf468f")
index = pinecone.Index(index_name)

def retrieve_documents(question):
    results = index.query(vector=model.encode([question])[0].tolist(), top_k=4)
    return [res["id"] for res in results.matches]


app = FastAPI()

@app.get("/query")
def generate(question: str):
    documentIDs = retrieve_documents(question)
    return {"answer": [f"DOCUMENT-ID: {records[id]['PMID']}\n FULL-AUTHOR: {records[id]['FAU']}\n PUBLICATION-DATE: {records[id]['DP']}\n TEXT: {records[id]['AB']}\n" for id in documentIDs]}


if __name__ == "__main__":
    uvicorn.run("pineconeEndpoint:app", port=5000, log_level="info")