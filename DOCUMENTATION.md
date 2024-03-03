# Medical Chatbot using finetuned LLM and RAG on Pubmed dataset

|                   | GitHub Handle | E-Mail                       | Course of Study           | Matriculation Number |
| ----------------- | ------------- | ---------------------------- | ------------------------- | -------------------- |
| Jonas Gann        | @J-Gann       | gann@stud.uni-heidelberg.de  | Data and Computer Science | 3367576              |
| Christian Teutsch | @chTeut       | christian.teutsch@outlook.de | Data and Computer Science | 3729420              |
| Saif Mandour      | @saifmandour  | saifmandour@gmail.com        | Computer Science          | 4189231              |

Advisor:

## Introduction

This text serves as documentation for a final project as part of the course "Natural Language Processing with Transformers" at Heidelberg University.
The objective as stated in the project description was to develop a domain-specific question-answering (QA) system either using a medical or legal domain. Given natural language input from the user, the system should be able to generate natural language answers based on relevant documents from a document corpus. The system should be capable of answering a variety of question types and also provide a user interface.

The focus of the project lies on leveraging large language models (LLMs) for context specific tasks. Although the increase in quality of large language models in the past years is astonishing in itself, it is always worthwhile to take a look at how their capabilities can be used to solve real-life problems. There are many examples of successful application of text-based generative models. Usually this is possible "out-of-the-box" however adaptation to a given use case can lead to better results. Common approaches are "finetuning" of existing models and context enrichment from relevant data-sources using "retrieval augmented generation" (RAG).

Often these approaches are used separately, however in this project we utilize a LLM finetuned on biomedical data in combination with a RAG system based on a corpus of medical scientific papers. As will be demonstrated later in this text, LLMs which are finetuned on a context can be nicely expanded by a RAG system of the same context, combining general context knowledge gained during finetuning with more specific knowledge gained through document retrieval by the RAG system.

In the following report we will first give an overview of existing work which is relevant for the project. After that we will give a detailed explanation of the developed system and its components. Finally, we will present the final results and give an outlook on possible improvements and future work.

## Related Work

This section presents previous work, relevant for our project. As this was a software development project with the goal of a "finished product", we heavily used and combined existing models and software. Therefore we will focus on the description of these tools and only briefly cover the underlying scientific work.

Lewis et al. Introduced the RAG approach for natural language processing (NLP) tasks in a specialized field like the medical topic of our project[^13]. RAG combines the strengths of information retrieval (IR) and generative models, allowing for more accurate and contextually relevant outputs. The system uses an IR component to retrieve relevant information from a vector database, which is then used to augment the input to a LLM.  
The idea of using vectors to represent documents is not new and was utilized in the context of information retrieval long before. Salton et al. introduced a vector space model for automatic indexing as early as 1975[^14]. The vector space model (VSM) represents documents in a high-dimensional space, with each dimension corresponding to a term in the vocabulary. The value in each vector represents the frequency of the corresponding term.  
The concept of representing words or documents as vectors has evolved over time. Tomas Mikolov et al. Introduced the Word2Vec algorithm, which is a neural network-based approach for learning word embeddings from large text corpora[^15]. The key idea behind Word2Vec is to train a neural network to predict the context of a word based on its surrounding words in a sentence. The resulting word embeddings capture the semantic relationships between words.  
Jeffrey Penning et al. introduced a similar approach in their paper[^16]. The key idea behind GloVe is to learn word embeddings by capturing the co-occurrence statistics of words in a corpus. The algorithm constructs a co-occurrence matrix that represents the frequency of word pairs occurring together in a context window. This matrix is then factorized to obtain word embeddings that capture the semantic relationships between words. 
These are two examples how to embed language into a vector representation. These word embeddings can be stored in the vector database like Pinecone[^5] or Opensearch[^6] where a similarity measure, such as cosine similarity or Euclidean distance, is used to compute the similarity between the query document and each document in the database.  A k-Nearest Neighbors (k-NN) algorithm is applied to the computed similarities to retrieve the k most similar documents to the query document. The k nearest neighbors are the documents with the highest similarity scores. 

Touvren et al. trained large transformers on large quantity of textual data and made the models publicly available to the research community[^17]. Transformers are an groundbraking architecture introduced by Ashish Vaswani et al.[^18]. The key innovation in the Transformer architecture is the use of self-attention mechanisms, which allow the model to focus on different parts of the input sequence when generating the output sequence. Parallel processing of the input sequence enabled the training of models on massive datasets, as demonstrated in[^17]. 
The availability of open-source models has encouraged the development of streamlined tools like Ollama[^8], designed to run these models locally. Ollama provided access to BioMistral, a model which was introduced by Morin et al.[^19]. BioMistral is based on Mistral, a 7-billion-parameter llm introduced by Jian et al.[^20]. Despite having fewer parameters, it outperforms high-performing language models like Llama 1 and Llama 2, which have 13 billion and 34 billion parameters, respectively. Morin et al. fine-tuned their model on the free-accessable PubMed Central database which contains wide range of scientific research articles from the MEDLINE database[^1]. The United States National Library of Medicine (NLM) at the National Institutes of Health (NIH) maintains the MEDLINE database as part of the Entrez system of information retrieval.  

To make our RAG system accessible for users, we take advantage of Hugging Face, an open-source community focused on NLP and machine learning (ML). Hugging Face offers a chat interface called Chat-UI, the source code is accessible and can be customized[^12]. Chat-UI is build by SvelteKit, a framework for building web applications and websites using the Svelte framework[^21]. It provides a range of features for building web applications, including a compiler that generates highly optimized JavaScript code, a component-based architecture, and a reactive data store. 

Another important technology we used is Docker, a platform for developing, shipping, and running applications using containerization[^22]. A Dockerfile contains instructions for building a Docker image. The Docker image is a standalone package that contains the application and all of its dependencies. An instance of the image can be run as a Docker container on a host machine. This process simplifies the running of an application due to the pre-definition of dependencies, which is a common problem when running an application on another host machine. 

## Methods and Approach

In this section we present the developed system in detail, discuss design choices and mention problems and pitfalls we noticed. We will describe the retrieval, preprocessing and storage of the data. We also go into detail about our utilization of pretrained LLMs, how we designed and incorporated a RAG system and how we finalized the system with a user interface. Finally we will briefly go over methods and tools we used during development.

### Overall Design Choices

In this project we focused on delivering a finished project which could actually be useful in its intended domain. Therefore we decided to not approach the project as a research project but as a software development project. This means we focused on the development of a system which is capable of answering questions in the medical domain. We did not focus on the development of new models or algorithms but rather on the integration of existing models and tools. We also focused on the possibility of future expansion of the code by using open-source tools and adding new features in an expandable way. With regards to the behaviour of the answer generation we favoured simplicity, consistency and reliability instead of the highest possible answer quality.

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
The alternative to a fully hosted service such as Pinecone is to use a locally running system like OpenSearch[^6]. It is highly costumizable and enables storing embeddings with the sources. The REST API makes is easy to query the top-k similar embeddings and because it runs locally it runs much faster than for document retrival and later references.

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

We decided to not write a custom user interface as there were many open-source user interfaces available for the use case at hand. We initially looked at user interfaces in the context of Ollama such as Open WebUI[^11]. However this turned out not to be sufficiently expandable in order to incorporate our RAG system. Therefore we decided to use the extensively customizable Chat-UI[^12].

Here we were able to not only customize the models available to the user in a configuration file but also insert custom code, adding new features to the answer generation pipeline. This most notably concerns the ["buildPrompt.ts"](./chat-ui-rag/src/lib/buildPrompt.ts) file in which the main logic regarding the enrichment of the user prompt is contained. Here we added the option to insert papers retrieved by the RAG system as described in the previous chapter. We added a "rag" configuration option to the [".env"](./chat-ui-rag/.env) file to enable the integrator to turn the RAG system on and off. Additionally we wanted the vector store to also be replacable as we saw no need to hardcode the usage of Pinecone. Thats why we added the option "vectorStoreType" as well as "url" to enable configuration of different vector stores. To use a different vector store one would have to implement an adapter for the new vector store and implement the corresponding endpoint similar to the ["PineconeEndpoint](./chat-ui-rag/src/lib/server/rag/pinecone/pineconeEndpoint.ts) and add a new case to the ["RAGEndpoint](./chat-ui-rag/src/lib/server/rag/ragEndpoint.ts). This way we provided two vector stores: Pinecone and OpenSearch.

![Welcome Page](evaluation/images/welcome.png)

![UI with answer](evaluation/images/prompt.png)

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

The data was retrieved as described in the Data Retrieval section. Each entry contains 20 different meta-information fields. We decided to use the unique PubMed identification PMID to identify our documents in the vector database. The abstract (AB) is used to generate the word embedding to calculate similarity between documents and the question query. We also provide the source (SO) by concatenating it with the first author's (FAU) name. Additionally, we save the title (TI) and the date of publication (DP).

We used t
- what data used for project (medline data format)
- when, how, where (already mentioned in Approach?)
- metrics of data

### Evaluation Method

We used three different evaluation settings. 

1. Generation of PubMed related questions and answers (QA's) 

The QA's pairs are related to the word "intelligence" in the abstract. We used Chat-GPT to generate both questions and answers. We used the answers for the same questions to compare the answers of our RAG system. To do that, we calculate the word embeddings of both answers and calculate the similarity. This is done with the Python scripts answerEmbedding.py and answerSimilarity.py from the /QA-INLPT-WS2023/evaluation folder. The information is stored in QAs.json, and the results are shown in Table ??. We also check the retrieval score and the grammar of the generated question of the RAG system.

2. Generated unrelated QA's 

We used QA's pairs that are unrelated to a medical context to assess how the BioMistral model, fine-tuned on PubMed data, performs. We are interested in whether the Q&A pairs found reasonable answers and if the RAG system found a source. The results are shown in Table ???.

3. Sentiment analysis 

We compared the sentiment analysis of questions that should be answered with either Yes (positive) or No (negative). The results are shown in Table ???.


- explain how performance is evaluated
- quantitative or qualitative

### Experimental Details
We used the follwing prompts to generate the QA's in Chat-GPT.  

1. Generation of PubMed related QA's

User: "Are you familiar with PubMed dataset?"
User: "Generate 20 questions about the topic of intelligence in a medical context which a medical assistant can answer. "
User: "You are a medical assistant. Answer questions truthfully. Base your answer solely on PubMed data."

2. Generated unrelated QA's

User: "Are you familiar with PubMed dataset?"
User: "Generate 5 question which has nothing to do with PubMed or any other medical topic."
User: "Can you answer the 5 questions?"

3. Sentiment analysis

User: "Are you familiar with PubMed dataset?"
User: "Generate 20 questions about the topic of intelligence in a medical context which a medical assistant can answer. "
User: "You are a medical assistant. Answer questions truthfully. Base your answer solely on PubMed data."
User: "Can you proviide 10 questions and answers which have a positive or negative answer? It must be possible to answer with yes or no."


- used some configurable evaluation?


### Results

1. Generation of PubMed related QA's

We calculated the similarity of the answers for all questions. The retrieval score was logged together with the answers. Note that the unanswered questions are due to a key error. We updated the source information later, and it was not filtered when we created the database. So, it could be that a document from the vector database has the highest score but no source. If this is the case, we get an error in the pinceoneEndpoint.py function where we filter documents without a source. However, we decided to leave the results in the test data to do a countercheck. We plan to update the vector database before the submission deadline.

| Question  | Chat-GPT |  RAG    | Similarity | Score | Grammar |
| --------- | -------- | ------- | ---------- | ----- | ------- |
|     1     |  correct | correct |    0.80    | 0.69  |         |
|     2     |  correct | correct |    0.75    | 0.63  |         |
|     3     |  correct |   error |   -0.12    |       |         |
|     4     |  correct |   error |   -0.05    |       |         |
|     5     |  correct |   error |   -0.11    |       |         |
|     6     |  correct | correct?|    0.68    | 0.71  |         |
|     7     |  correct | correct?|    0.80    | 0.73  |         |
|     8     |  correct | correct?|    0.68    | 0.73  |         |
|     9     |  correct | correct?|    0.80    | 0.72  |         |
|    10     |  correct | correct?|    0.66    | 0.69  |         |
|    11     |  correct | correct?|    0.76    | 0.62  |         |
|    12     |  correct | correct?|    0.67    | 0.62  |         |
|    13     |  correct | correct?|    0.82    | 0.74  |         |
|    14     |  correct | correct?|    0.84    | 0.73  |         |
|    15     |  correct | correct?|    0.85    | 0.71  |         |
|    16     |  correct | correct?|    0.87    | 0.63  |         |
|    17     |  correct | correct?|    0.64    | 0.70  |         |
|    18     |  correct | correct?|    0.85    | 0.61  |         |
|    19     |  correct | correct?|    0.72    | 0.68  |         |
|    20     |  correct |   error |   -0.09    |       |         |
|    21     |  correct | correct?|    0.62    | 0.67  |         |
|    22     |  correct | correct?|    0.61    | 0.68  |         |
|    23     |  correct | correct?|    0.82    | 0.68  |         |

2. Generated unrelated QA's

| Question  | Chat-GPT |  RAG    | Similarity | Score | Grammar | Source |
| --------- | -------- | ------- | ---------- | ----- | ------- |--------|
|     1     | correct? | correct?|    0.94    | 0.48  |         |   Yes  |
|     2     | correct? | correct?|    0.79    | 0.43  |         |   Yes  |
|     3     | correct? | correct?|    0.86    | 0.36  |         |    No  |
|     4     | correct? | correct?|    0.83    | 0.26  |         |    No  |
|     5     | correct? | correct?|    0.95    | 0.53  |         |   Yes  |

3. Sentiment analysis

| Question  | Chat-GPT |  RAG    | Similarity | Score | Grammar | Source |
| --------- | -------- | ------- | ---------- | ----- | ------- |--------|
|     1     |    Yes   |    No   |    0.0     | 0.63  |         |   Yes  |
|     2     |    Yes   |   Yes   |    1.0     | 0.70  |         |   Yes  |
|     3     |    Yes   |   Yes   |    1.0     | 0.71  |         |   Yes  |
|     4     |    Yes   |    No   |    0.0     | 0.67  |         |   Yes  |
|     5     |    Yes   |  Unsure |    0.5     | 0.79  |         |   Yes  |
|     6     |     No   |    No   |    1.0     | 0.76  |         |   Yes  |
|     7     |    Yes   |   Yes   |    1.0     | 0.72  |         |   Yes  |
|     8     |     No   |  Unsure |    0.5     | 0.66  |         |   Yes  |
|     9     |     No   |    No   |    1.0     | 0.63  |         |   Yes  |
|    10     |     No   |    No   |    1.0     | 0.77  |         |   Yes  |

- baseline comparison?
- tables, plots?
- expectations vs. reality?

### Analysis

1. Generation of PubMed related QA's

       1. Question: Both answers contain the words problem-solving, resoning and ability to learn. However RAG mentioned that there is no definition of intelligence in medical context. Chat-GPT defines it in the answers.
       2. Question: The answer of RAG referes to measure of a intelligence in a admission process of a medical school. Chat-GPT is more general and referes to measuere the intelligence with an IQ test.
       3. No RAG answer
       4. No RAG answer
       5. No RAG answer
       6. RAG cites the same source as in question 2. However, the text does not really summarize the abstract and seems that the source document is not really related to the question. It's more about if specific jobs in medicine need more intelligence which will be asked later. The Chat-GPT answer is short an precise. 
       7. Opposite answers. RAG states that it is possible to improve or maintain intelligence of children. Chat-GPT states that it is not possible to improve intelligence, only the maintain intelligence is possible by activation the cognitive function with games, reading etc.  
       8. RAG does not answer the question and just print the abstract of the highest scored document. However, the abstract is the right topic. It's about research of the change of intelligence by aging. The answer from Chat-GPT goes in the same direction but is shorter.
       

Why do we sometimes get the abstract as an answer???

Unrelated questions -> Lower retrieval score, higher similiarity.

How to handle documents which are published after February 2022 (Chat-GPT knowledge state)?

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
[^13]: Lewis, Perez, Piktus. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in Neural Information Processing Systems, 2020. 
       https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html
[^14]: Salton, Wong, Yang. A vector space model for automatic indexing. Communications of the ACM, 1975.
       https://dl.acm.org/doi/abs/10.1145/361219.361220 
[^15]: Mikolov, Chen, Corrado. Efficient estimation of word representations in vector space. arXiv preprint arXiv:1301.3781, 2013. 
       https://arxiv.org/abs/1301.3781 
[^16]: Pennington, Socher, Manning. Glove: Global vectors for word representation. Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP), 2014. 
       https://aclanthology.org/D14-1162.pdf 
[^17]: Touvron, Lavril, Izacard. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 
       https://arxiv.org/abs/2302.13971 
[^18]: Vaswani, Shazeer, Parmar. Attention is all you need. Advances in neural information processing systems, 2017. 
       https://proceedings.neurips.cc/paper/7181-attention-is-all
[^19]: Labrak, Bazoge, Morin. BioMistral: A Collection of Open-Source Pretrained Large Language Models for Medical Domains. arXiv preprint arXiv:2402.10373, 2023.
       https://arxiv.org/abs/2402.10373
[^20]: Jiang, Sablayrolles, Mensch. Mistral 7B. arXiv preprint arXiv:2310.06825, 2023.  
       https://arxiv.org/abs/2310.06825 
[^21]: https://kit.svelte.dev/ 
[^22]: https://www.docker.com/










