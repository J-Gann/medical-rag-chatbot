from dotenv import set_key
from pathlib import Path
import json


env_file_path = Path("/home/paperspace/QA-INLPT-WS2023/chat-ui-rag/.env")

data = [{ "name": "Biomistral RAG",
      "chatPromptTemplate": "<s>{{#each messages}}{{#ifUser}}[INST] {{#if @first}}{{#if @root.preprompt}}{{@root.preprompt}}\n{{/if}}{{/if}} {{content}} [/INST]{{/ifUser}}{{#ifAssistant}}{{content}}</s> {{/ifAssistant}}{{/each}}",
      "preprompt": "You are a medical assistant. Answer questions truthfully. Base your answer soly on the given information.",
      "websiteUrl": "https://huggingface.co/BioMistral/BioMistral-7B",
      "parameters": {
        "temperature": 1,
        "max_new_tokens": 1024,
        "stop": ["</s>"]
      },
      "rag": {
        "vectorStoreType": "pinecone",
        "url": "http://127.0.0.1:5000"
      },
      "endpoints": [
        {
         "type": "ollama",
         "url" : "http://127.0.0.1:11434",
         "ollamaName" : "cniongolo/biomistral"
        }
      ]
  }
]
  
models = json.dumps(data)

set_key(dotenv_path=env_file_path, key_to_set="MODELS", value_to_set=models)