import numpy as np
from sentence_transformers import SentenceTransformer

sentences:list = ["おはようございます"]

model = SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1')
embeddings:list = model.encode(sentences)
print(embeddings)

emb_1 = embeddings[0]
emb_2 = embeddings[1]

sim = np.dot(emb_1, emb_2) / (np.linalg.norm(emb_1) * np.linalg.norm(emb_2))
print(sim)