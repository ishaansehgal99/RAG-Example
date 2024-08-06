from sentence_transformers import SentenceTransformer
from tqdm import tqdm

model = SentenceTransformer('all-MiniLM-L6-v2')

def single_embedding(text):
    embedding = model.encode(text)
    return embedding


if __name__ == "__main__":
    texts = [
        "hello there", 
        "my old friend",
        'The dog is barking',
        'The cat is purring',
        'The bear is growling',
        'The dog is barking',
        'The cat is purring',
        'The bear is growling',
        'The dog is barking',
        'The cat is purring',
        'The bear is growling',
    ]
    for text in texts:
        print(single_embedding(text))
        