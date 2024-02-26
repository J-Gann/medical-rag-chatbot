import torch
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from Bio import Medline
import ollama
from fastapi import FastAPI
import uvicorn
from dotenv import set_key
from pathlib import Path
import json

env_file_path = Path("/home/paperspace/QA-INLPT-WS2023/chat-ui-rag/.env")

data = [{ "name": "Biomistral RAG",
      "chatPromptTemplate": "<s>{{#each messages}}{{#ifUser}}[INST] {{#if @first}}{{#if @root.preprompt}}{{@root.preprompt}}\n{{/if}}{{/if}} {{content}} [/INST]{{/ifUser}}{{#ifAssistant}}{{content}}</s> {{/ifAssistant}}{{/each}}",
      "preprompt": "You are a medical assistant. Answer questions truthfully. Base your answer soly on the given information.",
      "websiteUrl": "https://huggingface.co/BioMistral/BioMistral-7B",
      "parameters": {
        "temperature": 1,
        "max_new_tokens": 1024,
        "stop": ["</s>"]
      },
      "rag": {
        "vectorStoreType": "pinecone",
        "url": "http://127.0.0.1:5000"
      },
      "endpoints": [
        {
         "type": "ollama",
         "url" : "http://127.0.0.1:11434",
         "ollamaName" : "cniongolo/biomistral"
        }
      ]
  }
]
  
models = json.dumps(data)

set_key(dotenv_path=env_file_path, key_to_set="MODELS", value_to_set=models)



if torch.cuda.is_available():
  device = "cuda:0"
else:
  device = "cpu"
device

records = {}
missed = 0


PUBMED_PATH = "/home/paperspace/QA-INLPT-WS2023/chat-ui-rag/src/lib/server/rag/pinecone/pubmed_data"
with open(PUBMED_PATH) as stream:
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

        if not "SO" in article:
            missed += 1
            continue
        
        
        records[article["PMID"]] = article

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

index_name = 'pubmed'
pinecone = Pinecone(api_key="7377f10c-e728-4408-b2a6-3ac28daf468f")
index = pinecone.Index(index_name)

def retrieve_documents(question):
    results = index.query(vector=model.encode([question])[0].tolist(), top_k=1)
    return [(res["id"],res["score"]) for res in results.matches]


app = FastAPI()

@app.get("/query")
def generate(question: str):
    documents = retrieve_documents(question)

    documentid, score = documents[0]

    text = records[documentid]['AB']

    source = "\n"+records[documentid]['FAU'][0]+" et al.\n"+records[documentid]['SO'].replace('doi: ', 'https://doi.org/')

    res = {"text": text, "source": source, "score": score}

    return res

if __name__ == "__main__":
    uvicorn.run("pineconeEndpoint:app", port=5000, log_level="info")
