# Medical Chatbot using finetuned LLM and RAG on Pubmed dataset

|                   | GitHub Handle | E-Mail                       | Studienfach               | Matrikelnummer |
| ----------------- | ------------- | ---------------------------- | ------------------------- | -------------- |
| Jonas Gann        | @J-Gann       | gann@stud.uni-heidelberg.de  | Data and Computer Science | 3367576        |
| Christian Teutsch | @chTeut       | christian.teutsch@outlook.de |                           |                |
| Saif Mandour      | @saifmandour  | saifmandour@gmail.com        |                           |                |

Advisor:

## Introduction

This text serves as documentation for a final project as part of the course "Natural Language Processing with Transformers" at Heidelberg University.
The objective as stated in the project description was to develop a domain-specific question-answering (QA) system either using a medical or legal domain. Given natural language input from the user, the system should be able to generate natural language answers based on relevant documents from a document corpus. The system should be capable of answering a variety of question types and also provide a user interface.

The focus of the project lies on leveraging large language models (LLMs) for context specific tasks. Although the increase in quality of large language models in the past years is astonishing in itself, it is always worthwhile to take a look at how their capabilities can be used to solve real-life problems. There are many examples of successful application of text-based generative models. Usually this is possible "out-of-the-box" however adaptation to a given use case can lead to better results. Common approaches are "finetuning" of existing models and context enrichment from relevant data-sources using "retrieval augmented generation" (RAG).

Often these approaches are used separately, however in this project we utilize a LLM finetuned on biomedical data in combination with a RAG system based on a corpus of medical scientific papers. As will be demonstrated later in this text, LLMs which are finetuned on a context can be nicely expanded by a RAG system of the same context, combining general context knowledge gained during finetuning with more specific knowledge gained trough document retrieval by the RAG system.

In the following report we will first give an overview of existing work which is relevant for the project. After that we will give a detailed explanation of the developed system and its components. Finally, we will present the final results and give an outlook on possible improvements and future work.

## Related Work

This section presents previous work, relevant for our project. As this was a software development project with the goal of a "finished product", we heavily used and combined existing models and software. Therefore we will focus on the description of these tools and only briefly cover the underlying scientific work.

### Ollama

### Biomistral

### Chat-Ui

### PubMed

## Methods and Approach

In this section we present the developed system in detail, discuss design choices and mention problems and pitfalls we noticed. We will describe the retrieval, preprocessing and storage of the data. We also go into detail about our utilization of pretrained LLMs, how we designed and incorporated a RAG system and how we finalized the system with a user interface. Finally we will briefly go over methods and tools we used during development.

### Data Retrieval

We decided to focus on the medical domain for this project. As part of the project description we were guided to use data from PubMed[^1] and limit ourselves to papers containing the word "intelligence" in either the abstract or the title and a publishing date between 2013 and 2023. At the time of writing, this resulted in 88.486 papers. We initially tried to download the relevant papers using Biopython[^2] but noticed a datacap of 10.000 papers, which was well below our needs. We then discovered EDirect[^3] [^4], a unix command line utility covering the same functionality but without the datacap. Instead of the standard XML format we decided to use Medline as it turned out to be more convenient for us to work with and required less disk space. We used the following command to download the relevant papers.

```shell
./esearch -db pubmed -query "intelligence[tiab]" -mindate 2013 -maxdate 2023 | ./efetch -format medline > ./pubmed_data
```

### Data Preprocessing

- used title, abstract and author
- what were problems?

### Data Storage

- Vectorstore: Pinecone / Opensearch
- Datastorage: No database / in-memory
- what were problems?

### LLM Model

- Ollama (other: HuggingFace, LangChain)
- BioMistral (other: Mistral, LLama2)
- Prompt Engineering
- Performance
- Quirks and annoyances

### RAG System

- embedding of whole abstracts
- retrieval of most relevant paper (use only if score sufficiently high)
- text + metadata infused into prompt
- previous questions (and answers?) inserted into user prompt
- prompt engineering?
- what were problems?

### Document Reference

- source and url inserted into model answer
- other approaches error prone
  - prompt engineering

### User Interface

- chat-ui (other: ollama based, custom ui)
- what are cool features?
  - pretty ui
  - expandable
  - customizable
  - open source
- what were problems?
  -

### Deployment

- did not containerize
- why?
- etc.

### Development

- Paperspace
- Google Colab
- GitHub

## Collaboration

- Regular meetings
- What worked, what did not?
- ...

## Experiments

### Data

- what data used for project (medline data format)
- when, how, where (already mentioned in Approach?)
- metrics of data

### Evaluation Method

- explain how performance is evaluated
- quantitative or qualitative

### Experimental Details

- used some configurable evaluation?

### Results

- baseline comparison?
- tables, plots?
- expectations vs. reality?

### Analysis

- qualitative analysis
- consistency?
- surprising fails?
- baseline?
- show examples or numbers

## Contributions

|                   | Data Retrieval | Data Preprocessing | Data Storage | LLM Model | RAG System | Document Reference | User Interface | Experiments |
| ----------------- | -------------- | ------------------ | ------------ | --------- | ---------- | ------------------ | -------------- | ----------- |
| Jonas Gann        |                |                    |              |           |            |                    |                |             |
| Christian Teutsch |                |                    |              |           |            |                    |                |             |
| Saif Mandour      |                |                    |              |           |            |                    |                |             |

### Jonas Gann

### Christian Teutsch

### Saif Mandour

## Conclusion and Future Work

- recap main contributions
- reflect on limitations
- possible improvements / expansions
- what have we learned?

## Anti-plagiarism Confirmation

[^1]: https://pubmed.ncbi.nlm.nih.gov/
[^2]: https://biopython.org/
[^3]: https://www.nlm.nih.gov/dataguide/edirect/install.html
[^4]: https://www.ncbi.nlm.nih.gov/books/NBK179288/
