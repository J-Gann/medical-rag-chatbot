# Project Development Notes

- decided to use medical dataset
- researched on methods to download and use data in python
  - https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/
  - https://pypi.org/project/pymed/
  - https://medium.com/@felipe.odorcyk/scrapping-data-from-pubmed-database-78a9b53de8ca
  - https://www.youtube.com/watch?v=d_juHx8KE8k
- finally decided to go with Entrez from Biopython
  - discovered download limit of 10000 articles
  - decided 10000 articeles are enough for the moment
  - TODO: download more articles later using https://www.nlm.nih.gov/dataguide/edirect/install.html#edirect-installation
- created google colab project
- made some basic preprocessing (extract document id, abstract and title)
  - TODO: utilize much more metadata in the future
- generate embeddings for documents and questions to filter for relevant documents
- search for relevant documents using cosine similarity

- started to implement a simple question answering system
- summarize abstracts of relevant documents to reduce prompt to reasonable size
- use pretrained question answering model

  - generates mostly correct answers but answers are very short (few words)

- Try different approach by extracting most relevant sentences from relevant documents and summarize them to get longer answers

  - Leads to better results. Sentences are longer and contain relevant information. However sentences are imperfect regarding grammar and punctuation. Content of sentences is also a bit erratic

- Try to use more powerful Large Language Model for Natural Question Answering
- Try to use Mixtral 8x7B instruct
  - way too large to use on our colab subscription
- Try to use LLama 2

  - works very good. Answers are mostly correct and long enough. However, it is quite slow.
  - also, we were not yet able to utilize the colab GPU, as the model is too large to fit into the GPU memory

- First eperiments to add citation feature by expanding the prompt for LLama
  - Works quite well, however the model tries to include a citation for every model resulting in something like a summarization of each document in the context

- Take a look at paperspace as an alternative to colab
  - paperspace allows to use local dev environment (VsCode) which is a huge benefit
