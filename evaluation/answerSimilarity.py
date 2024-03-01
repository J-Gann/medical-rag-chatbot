import spacy

nlp = spacy.load("en_core_web_sm")

# Generated answers 
chat_ui_answer= "..."
chat_gpt_answer = "The neurobiological mechanisms underlying intelligence are complex and involve multiple brain regions, including the prefrontal cortex, parietal cortex, and hippocampus. These regions are involved in various cognitive functions such as memory, attention, and problem-solving."

# answer processing 
answer1 = nlp(chat_ui_answer)
answer2 = nlp(chat_gpt_answer)

# Answer similarity 
similarity = answer1.similarity(answer2)

print(f"Similarity between the two sentences: {similarity}")
