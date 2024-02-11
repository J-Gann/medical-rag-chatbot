import ollama
import bs4
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, ReadTheDocsLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
import json


pubmed_data_path = "/home/chris/University/NLP_project/pubmed_data.json"
pubmed_data_preprocessed_path = "/home/chris/University/NLP_project/pubmed_data_preprocessed.json"

if not os.path.exists(pubmed_data_preprocessed_path):
  with open(pubmed_data_path, 'r') as f:
    records = json.loads(f.read())
     
  records = records["PubmedArticle"]
  preprocessed_records = []
  for idx, article in enumerate(records):
      if (not "Abstract" in article["MedlineCitation"]["Article"].keys()): continue
      article = {
          "_id": article["MedlineCitation"]["PMID"],
          "title": article["MedlineCitation"]["Article"]["ArticleTitle"],
          "text": " ".join(article["MedlineCitation"]["Article"]["Abstract"]["AbstractText"]) # some abstracts are split in an array
      }
      preprocessed_records.append(article)
  with open(pubmed_data_preprocessed_path, 'w') as f:
    f.write(json.dumps(preprocessed_records))
else:
    with open(pubmed_data_preprocessed_path, 'r') as f:
        preprocessed_records = json.loads(f.read())

text_list = [record['text'] for record in preprocessed_records]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=5)
splits = [text_splitter.split_text(str(text)) for text in text_list]


# Create Ollama embeddings and vector store
embeddings = OllamaEmbeddings(model="mistral")
vectorstore = Chroma.from_texts(texts=splits, embedding=embeddings)

# Create the retriever
retriever = vectorstore.as_retriever()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Define the Ollama LLM function
def ollama_llm(question, context):
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    response = ollama.chat(model='mistral', messages=[{'role': 'user', 'content': formatted_prompt}])
    return response['message']['content']

# Define the RAG chain
def rag_chain(question):
    retrieved_docs = retriever.invoke(question)
    formatted_context = format_docs(retrieved_docs)
    return ollama_llm(question, formatted_context)

# Use the RAG chain
result = rag_chain("What is Mistral?")
print(result)