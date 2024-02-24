import torch
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from Bio import Medline
import ollama
from fastapi import FastAPI
import uvicorn

from opensearchpy import OpenSearch
from transformers import AutoTokenizer, AutoModel, BertForSequenceClassification, AutoModelForQuestionAnswering



if torch.cuda.is_available():
  device = "cuda:0"
else:
  device = "cpu"
device

records = {}
missed = 0

"""
Add the whole path if pubmed_data is not detectet
e.g. /home/paperspace/QA-INLPT-WS2023/chat-ui-rag/src/lib/server/rag/pinecone/pubmed_data
"""

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

#model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2').to(device)

# why not take cls token?
def mean_pooling(last_hidden_state, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return torch.sum(last_hidden_state * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# https://opensearch.org/docs/latest/clients/python-low-level/

host = 'localhost'
port = 9200
auth = ('admin', 'admin')

# Create the client with SSL/TLS enabled, but hostname verification disabled.
client = OpenSearch(
    hosts = [{'host': host, 'port': port}],
    http_compress = True, # enables gzip compression for request bodies
    http_auth = auth,
    use_ssl = True,
    verify_certs = False,
    ssl_assert_hostname = False,
    ssl_show_warn = False,
)

index_name = 'pub_med_index'

def retrieve_documents(question):
    
    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True).to(device)
    query_outputs = mean_pooling(model(**inputs).last_hidden_state, inputs["attention_mask"]).to("cpu")
    print(len(query_outputs[0].tolist()))

    # Define the KNN search query
    knn_query = {
        "size": 1,
        "_source": ["title", "text"],
        "query": {
            "knn": {
                "vector": {
                    "vector": query_outputs[0].tolist(),
                    "k": 1
                }
            }
        }
    }

    # Perform the KNN search
    response = client.search(index=index_name, body=knn_query)
    
    print(response)
    
    return [(res['_id'], res['_score']) for res in response['hits']['hits'][:]]
    
app = FastAPI()

@app.get("/query")
def generate(question: str):
    documents = retrieve_documents(question)
    return {"answer": [f"DOCUMENT-ID: {records[id]['PMID']}\n FULL-AUTHOR: {records[id]['FAU']}\n PUBLICATION-DATE: {records[id]['DP']}\n TEXT: {records[id]['AB']}\n SCORE: {round(score,2)} \n DOCUMENT-TITLE: {records[id]['TI']}" for id,score in documents]}


if __name__ == "__main__":
    uvicorn.run("opensearchEndpoint:app", port=9200, log_level="info")