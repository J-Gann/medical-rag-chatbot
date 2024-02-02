import torch
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from Bio import Medline
import ollama
from fastapi import FastAPI

if torch.cuda.is_available():
  device = "cuda:0"
else:
  device = "cpu"
device

records = {}
missed = 0

with open("pubmed_data") as stream:
    for article in Medline.parse(stream):

        if not "PMID" in article:
            missed += 1
            continue

        if not "TI" in article:
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
    results = index.query(vector=model.encode([question])[0].tolist(), top_k=5)
    return [res["id"] for res in results.matches]


def generate_prompt(question):

    context_ids = retrieve_documents(question)

    context = ""
    for id in context_ids:
        text = records[id]["AB"]
        context += f"""
DOCUMENT_ID: {id}
{text}
"""


    prompt = f"""ANSWER the following QUESTION soley based on the CONTEXT given. Use only CONTEXT relevant for the QUESTION. Keep your ANSWER as short and concise as possible. Mention the DOCUMENT-ID where appropriate. 

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    return prompt

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/generate")
def generate(question: str):
    prompt = generate_prompt(question)
    answer = ollama.generate(model="llama2", prompt=prompt)["response"]
    return {"answer": answer}
