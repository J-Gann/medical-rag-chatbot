import gradio as gr
import bs4
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import ollama
import os
import json

# Function to load, split, and retrieve documents
def load_and_retrieve_docs():
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

    text_list = [record['text'] for record in preprocessed_records[0:2]]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=5)
    splits = [text_splitter.split_text(str(text)) for text in text_list]

    print("splits")

    class MyDocument:
        def __init__(self, text, metadata=None):
            self.text = text
            self.metadata = metadata
            self.page_content = text # add this line to define page_content attribute

    metadata = [{'source': str(i)} for i in range(len(text_list))]
    documents = [MyDocument(text, metadata=metadata[i]) for i, text in enumerate(text_list)]

    # Create Ollama embeddings and vector store
    embeddings = OllamaEmbeddings(model="orca-mini")
    print("embeddings")

    #vectorstore = Chroma.from_texts(texts=splits, embedding=embeddings)
    vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings)

    print("vectorstore")

    # Create the retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    
    return retriever

# Function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Function that defines the RAG chain
def rag_chain(question):
    retriever = load_and_retrieve_docs()
    #retrieved_docs = retriever.invoke(question)
    retrieved_docs = retriever.get_relevant_documents(question)
    formatted_context = format_docs(retrieved_docs)
    formatted_prompt = f"Question: {question}\n\nContext: {formatted_context}"
    response = ollama.chat(model='mistral', messages=[{'role': 'user', 'content': formatted_prompt}])
    return response['message']['content']

# Gradio interface
iface = gr.Interface(
    fn=rag_chain,
    inputs="text",
    outputs="text",
    title="RAG Chain Question Answering",
    description="Enter a URL and a query to get answers from the RAG chain."
)

# Launch the app
iface.launch()