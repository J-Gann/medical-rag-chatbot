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
