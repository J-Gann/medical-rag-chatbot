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

Often these approaches are used separately, however in this project we utilize a LLM finetuned on biomedical data in combination with a RAG system based on a corpus of medical scientific papers. As will be demonstrated later in this text, LLMs which are finetuned on a context can be nicely expanded by a RAG system of the same context, combining general context knowledge gained during finetuning with more specific knowledge gained through document retrieval by the RAG system.

In the following report we will first give an overview of existing work which is relevant for the project. After that we will give a detailed explanation of the developed system and its components. Finally, we will present the final results and give an outlook on possible improvements and future work.

## Related Work

This section presents previous work, relevant for our project. As this was a software development project with the goal of a "finished product", we heavily used and combined existing models and software. Therefore we will focus on the description of these tools and only briefly cover the underlying scientific work.

### Embeddings

### Ollama

### Biomistral

### Chat-Ui

### PubMed

## Methods and Approach

In this section we present the developed system in detail, discuss design choices and mention problems and pitfalls we noticed. We will describe the retrieval, preprocessing and storage of the data. We also go into detail about our utilization of pretrained LLMs, how we designed and incorporated a RAG system and how we finalized the system with a user interface. Finally we will briefly go over methods and tools we used during development.

### Overall Design Choices

### Data Retrieval

We decided to focus on the medical domain for this project. As part of the project description we were guided to use data from PubMed[^1] and limit ourselves to papers containing the word "intelligence" in either the abstract or the title and a publishing date between 2013 and 2023. At the time of writing, this resulted in 63.508 papers. We initially tried to download the relevant papers using Biopython[^2] but noticed a datacap of 10.000 papers, which was well below our needs. We then discovered EDirect[^3] [^4], a unix command line utility covering the same functionality but without the datacap. Instead of the standard XML format we decided to use Medline as it turned out to be more convenient for us to work with and required less disk space. We used the following command to download the relevant papers.

```shell
./esearch -db pubmed -query "intelligence[tiab]" -mindate 2013 -maxdate 2023 | ./efetch -format medline > ./pubmed_data
```

### Data Preprocessing

As mentioned in the previous section, we already filtered the data to be published between the year 2013 and 2023. Additionally we limit the content of the papers to be about intelligence by filtering all papers which do not contain the term "intelligence" either in the title or in the abstract. We also require papers to contain at least the following data, which we later use in the system: ID, Title, Author, Publication-Date, Abstract, Source. Out of the 63.508 papers, 4.339 do not meet these requirements, resulting in a final dataset of 59.169 papers.

### Vector Storage

In order to query relevant papers for a given user question, we decided to compute an embedding for each paper. These embedding store the semantic content of a paper in form of a vector. One can calculate the cosine similarity between embeddings of the paper-abstracts and the embedding of a user question to find papers with similar content to the question. To further improve the semantic content of the embedding, one could not only incorporate the abstract of the paper into the embedding but also relevant metadata such as the names of the authors as well as the publication date.
Once ambeddings are pre-computed for all papers, they can be used to find relevant papers for all coming user questions by computing their embedding similarity. There exist specialized software tools which make these computations easy and efficient. One of the so called "vectorstores" is Pinecone[^5]. It enables cloud storage of embeddings and provides an API for easily querying the top-k most similar embeddings for a given user question.
The alternative to a fully hosted service such as Pinecone is to use a locally running system like OpenSearch[^6].

<!-- MORE about OpenSearch -->

### Data Storage

As the vector storage is only used to store the embedding values, we need a way to load the dataset containing all the abstracts and metadata. The dataset with 59.169 papers has a total of 260 MB. We decided to not use a database but to write a REST server, which loads the whole dataset into memory on startup and stores the papers in a dictionary with the paper ID as key. Once started, the server hosts an endpoint which consumes a string as input, and responds with the content of the most similar paper. We decided to implement this functionality as a server in order to be able to implement this feature using Python but also enable its usage in the user interface which is written in TypeScript.

### LLM Model

As we want to develop a system capable of understanding and generating natural language, we use a large language model at the hear of our system. We tried many different models and hosting solutions and found out, that BioMistral[^7] in combination with Ollama[^8] worked best in our case. Ollama proved to be very easy to install and setup and due to its quantized models answer generation was very efficient. We also tried to develop a "plug-and-play" solution for HuggingFace models using LangChain[^9] and LangServe[^10] but were challenged with severe problems due to lack of documentation of LangChain and LangServe as well as poor answer quality due to some misbehavior in the interactions of HuggingFace models and LangChain pipelines.

In the context of Ollama we tried a variety of models for answer generation. Well known models like Llama 2 and Mistral worked reasonably well but were difficult to adapt to our use case, as they tended to start and end by repeating colloquial phrases. We did not find suitable model configuration parameters or system prompts to alleviate this problem. Another issue was, that in case no suitable paper was found by the RAG system, these models were not able to generate answers of the same quality.

This was the reason why we decided to search for language models which were finetuned in medical context in the hope that they will be able to bridge the gap of cases where the RAG system does fail to find relevant papers. We also hoped, that such models also reflect the precise and professional writing style from scientific papers in the formulation of their answers.

We found BioMistral to be a perfect fit and fulfill all our expectations. Due to the finetuning, the model is able to accurately and professionally answer medical questions on its own as well as competently include additionally information provided by the RAG system.

We eventually settled for the following system prompt for the BioMistral model:

```js
"You are a medical assistant. Answer questions truthfully. Base your answer soly on the given information.";
```

### RAG System

Main part of the RAG system is the retrieval of relevant papers for a given user question, which was already introduced as part of the section "Vector Storage". Next to the paper content, the server also includes the similarity score in its response. In case the similarity score is higher than 0.5 (a value we defined after some experiments) the abstract of the paper is inserted into the user prompt as follows:

```js
`=====================

CONTEXT:

${abstract}

=====================

${userQuestion}`;
```

If the score is below 0.5, the string "No context available." is inserted instead.

We made the design choice to only retrieve and use the most similar paper. The reason is, that in case of multiple papers being inserted in the prompt, we were not able to instruct the llm model to generate a self-contained answer which met our expectations. Instead the answers would usually be a list of summaries of each paper abstract. We were not satisfied with this behavior and decided a consistent answer based on one paper was more appropriate than a list of summaries of relevant papers.

As we wanted the model to be aware of the chat history we included the previous user questions into the prompt:

```js
`=====================
  
${previousQuestions}

CONTEXT:

${abstract}

=====================

${userQuestion}`;
```

This enables the model to better understand the intention of the current question of the user. An interesting extension would be to also include previous answers of the model. In that case one would have to make sure, that the prompt does not get too large.

As we only use at most one paper for a question, we were able to manually append the source of the paper to the answer of the model. We experimented with different prompt statements instructing the model to automatically insert a reference to the paper where appropriate but did not find a solution which worked consistently. It would be very interesting to go through current research and incorporate promising approaches into the system.

### User Interface

We decided to not write a custom user interface as there were many open-source user interfaces available for the use case at hand. We initially looked at user interfaces in the context of Ollama such as Open WebUI"[^11]. However this turned out not to be sufficiently expandable in order to incorporate our RAG system. Therefore we decided to use the extensively customizable Chat-UI[^12].

Here we were able to not only customize the models available to the user in a configuration file but also insert custom code, adding new features to the answer generation pipeline. This most notably concerns the "buildPrompt.ts" file in which the main logic regarding the enrichment of the user prompt is contained. Here we added the configurable capability to insert papers retrieved by the RAG system as described in the previous chapter.

<!-- INSERT ui screenshots -->

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
[^5]: https://www.pinecone.io
[^6]: https://opensearch.org/
[^7]: https://huggingface.co/BioMistral/BioMistral-7B
[^8]: https://ollama.com/
[^9]: https://www.langchain.com/
[^10]: https://python.langchain.com/docs/langserve
[^11]: https://github.com/open-webui/open-webui
[^12]: https://github.com/huggingface/chat-ui
