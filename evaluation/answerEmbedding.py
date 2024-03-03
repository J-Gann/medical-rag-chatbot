import json
import torch
from transformers import AutoTokenizer, AutoModel

if torch.cuda.is_available():
  device = "cuda:0"
else:
  device = "cpu"

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2').to(device)

def mean_pooling(last_hidden_state, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return torch.sum(last_hidden_state * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Load the JSON data
with open('QAs.json', 'r') as file:
    data = json.load(file)


embedding1 = []
embedding2 = []
with torch.no_grad():
    for i in range(len(data)):
        answer1 = data[str(i+1)]['answers']['CHAT-GPT']
        answer2 = data[str(i+1)]['answers']['RAG']
        
        inputs = tokenizer([answer1,answer2], return_tensors="pt", padding=True, truncation=True).to(device)
        out = model(**inputs)
        pooled = mean_pooling(out.last_hidden_state, inputs["attention_mask"]).to("cpu")
            
        #print(pooled)
        #break
        data[str(i+1)]['embeddings']['CHAT-GPT'] = str(pooled.numpy()[0].tolist())
        data[str(i+1)]['embeddings']['RAG'] = str(pooled.numpy()[1].tolist())
       

# Save the updated JSON data
with open('QAs.json', 'w') as file:
    json.dump(data, file, indent=4)
