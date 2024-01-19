# Meeting Notes

## 23.11.23

### Selection of dataset

- What are key metrics for choosing a dataset
  - Ease of downloading the Data (collect resources)
  - Creating a structure of the downloaded dataset (consindering Opensearch)
  - Performance evaluation of the model
  - Understanding the format of the articles and amount of preprocessing necessary to work with data.
- TODO: Everyone takes a look at both datasets and evaluates them regarding these metrics

### Meeting Schedule

- We want to do weekly meetings on Thursday at 18:00 (1h)
  - Topics:
    - Check progress
    - Ask questions
    - Plan ahead

### Other

- Data acquesition for PubMed: https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/
- Python tool for PubMed: https://pypi.org/project/pymed/
- Scraping PubMed: https://medium.com/@felipe.odorcyk/scrapping-data-from-pubmed-database-78a9b53de8ca

## 18.12.23

- Meeting with tutor
- Presented current state of project
- Already made good progress for first milestone

## 12.01.24

- Talked about the current state of the project 
- Get familar with opensearch 
- Thoughts about how to implement a GUI (e.g. Qt)

## 14.01.24

- Preparing for tutor meeting next day
- Exchange research and development progress of group members
- Collect outstanding ToDo´s
  - incorporate all relevant articles (through the Biopython pip module we can only download 10000 articles. Using the linux cli, there is no limit, see the notes in our GitHub project)
  - use a proper database to manage larger dataset (also use out of the box k-NN algorithm for embedding similarity search)
  - extend the usage of article metadata (atm I only use the title, article id and abstract)
  - improve the citation mechanism
  - improve the prompt design of the llama 2 LLM
  - maybe experiment with other LLMs
  - design the user interface (Website + ollama?)
- Collect open questions

## 15.01.24

### Tutor Meeting

- presentation of group progress
- feedback from tutor:
  - use vector database -> Chris + Saif working on it
  - performance benefits from ollama for llama 2
  - take a look at LangChain
  - take a look at laama model temperature
  - tips regarding llama prompt: "answer short and concise"
  - remember that we will eventually have to create a test dataset
  - we may eventually have to create a 15 minute presentation video of our system

### Group Recap

- recap tutor meeting
- collect next steps and ToDo´s
  - Saif + Chris
    - Evaluate if LangChain (https://www.langchain.com/) is a good choice for us
    - get opensearch pipeline with small dataset working (population + retrieval)
    - create large dataset
      - use bash cli: https://www.nlm.nih.gov/dataguide/edirect/install.html#edirect-installation
    - preprocess data
      - filter improtant metadata
      - sanitize abstract text
    - find hosting solution for opensearch 
    - integrate hosted opensearch into project notebook
      1. retrieval of documents
      2. similarity measurement of embeddings using k-NN
  
  - Jonas
    - evaluate which metada is relevant and where to use / integrate it -> resulting preprocessing done by Chris and Saif
    - extend notebook code to utilize more metadata
    - work on prompting of llama
    - improve citation mechanism
    - improve performance of llama (ollama, llama.cpp)
    - experiment with model temperature
    - find solution for deployment of model (ollama, paperspace, huggingface, ...)
  
  - Open
    - GUI
    - creation of test dataset
    - 15 minute video presenting final system
    - project report
    - GitHub readme

## 19.01.24

- Opensearch 
  - Implemented bulk load in opensearchpy and validated data in opensearch-dashboard  
  - Getting familar with opensearch analyzer for document retrieving

