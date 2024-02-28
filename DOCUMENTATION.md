# Medical Chatbot using finetuned LLM and RAG on Pubmed dataset

|                   | GitHub Handle | E-Mail                       | Studienfach               | Matrikelnummer |
| ----------------- | ------------- | ---------------------------- | ------------------------- | -------------- |
| Jonas Gann        | @J-Gann       | gann@stud.uni-heidelberg.de  | Data and Computer Science | 3367576        |
| Christian Teutsch | @chTeut       | christian.teutsch@outlook.de |                           |                |
| Saif Mandour      | @saifmandour  | saifmandour@gmail.com        |                           |                |

## Introduction

- Project Description
- Why is this Project relevant / interesting?
- Outline of report
- Key ideas of approach
- Outlook on results

## Related Work

- What is prior work we rely on?
- What is the context of our work?
- What are papers we relied on? What did we do different?

## Methods and Approach

- Conceptual details of the system
- Data preprocessing
- Developed algorithms
- Methods used
- Systems used
- Be detailed, use figures
- Express what is good about our approach
- Baseline approaches?

### Data Retrieval

- show used script
- pip package had limitations
- used shell script
- used medline dataformat => easy for processing, consumes less space
- what were problems?

### Data Preprocessing

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

## Conclusion and Future Work

- recap main contributions
- reflect on limitations
- possible improvements / expansions
- what have we learned?

## References
