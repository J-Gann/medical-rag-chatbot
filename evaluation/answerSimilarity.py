import json
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

if torch.cuda.is_available():
  device = "cuda:0"
else:
  device = "cpu"

# Load the JSON data
with open('QAs.json', 'r') as file:
    data = json.load(file)

for i in range(len(data)):
    
    embedding1 = np.array(data[str(i+1)]['embeddings']['CHAT-GPT'][1:-1].split(','), dtype=float)
    embedding2 = np.array(data[str(i+1)]['embeddings']['RAG'][1:-1].split(','), dtype=float)
    
    embedding1 = embedding1.reshape(1, -1)
    embedding2 = embedding2.reshape(1, -1)
    
    data[str(i+1)]['similarity'] = cosine_similarity(embedding1, embedding2)[0][0]
        
# Save the updated JSON data
with open('QAs.json', 'w') as file:
    json.dump(data, file, indent=4)
