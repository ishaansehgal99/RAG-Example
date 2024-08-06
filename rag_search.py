import numpy as np

from embeddings import single_embedding

# Default embedding dimension used by SentenceTransformer('all-MiniLM-L6-v2')
EMBEDDING_DIM = 384 

class RAGSearch:
    def __init__(self):
        """Initialize the RAGSearch class and set up the faiss database."""
        print("Initializing RAG Search")
        self.data = [
            'Introduction to Artificial Intelligence and Machine Learning',
            'Supervised Learning Algorithms: Linear Regression and Logistic Regression',
            'Unsupervised Learning: Clustering Algorithms - KMeans and DBSCAN',
            'Neural Networks and Deep Learning: Introduction to TensorFlow and Keras',
            'Natural Language Processing (NLP) Techniques: Tokenization and Word Embeddings',
            'Computer Vision and Image Processing: Convolutional Neural Networks (CNNs)',
            'Reinforcement Learning and Decision Making: Q-Learning and Markov Decision Processes',
            'Time Series Forecasting: ARIMA and Exponential Smoothing Methods',
            'Model Evaluation and Performance Metrics: ROC Curves and Confusion Matrices',
            'Feature Engineering and Data Preprocessing: Handling Missing Data and Outliers', 
            "I went to the dentist the other day and he let me pick a prize out of the prize box",
            "He loves to play basketball",
            "Bailey wishes he had a nicer car",
            "She decided to take the bus because it was supposed to be quicker, but then it ended up being twenty minutes late."
            "Back in the day, I was a checker-playing champion"
            "I just saw the most eccentric brick",
            "We need congressional support",
            "Nobody is out there trying to be happy",
            "When they first met, they didn\'t get along",
            "This will make a real difference",
            "In city after city, big public meetings were held",
            "This is my big break",
            "Her oven had a habit of setting off the fire alarm for no reason",
            "Many trees are in the forest",
            "I'm not really a TV watcher",
            "I think I could fall asleep really quickly",
            "God, just shut up already",
            "They were the nation's biggest business",
            "No woman would buy that",
            "I like furry animals",
            "Back in the day, I was a checker-playing champion",
            "I just saw the most eccentric brick",
            "We need congressional support",
            "Nobody is out there trying to be happy",
            "When they first met, they didn\'t get along",
            "This will make a real difference",
            "In city after city, big public meetings were held",
            "This is my big break",
            "Her oven had a habit of setting off the fire alarm for no reason",
            "Many trees are in the forest",
            "I'm not really a TV watcher",
            "I think I could fall asleep really quickly",
            "God, just shut up already",
            "They were the nation's biggest business",
            "No woman would buy that",
            "I like furry animals",
            "I just saw the most eccentric brick",
            "We need congressional support",
            "Nobody is out there trying to be happy",
            "When they first met, they didn\'t get along",
            "This will make a real difference",
            "In city after city, big public meetings were held",
            "This is my big break",
            "Her oven had a habit of setting off the fire alarm for no reason",
            "Many trees are in the forest",
            "I'm not really a TV watcher",
            "I think I could fall asleep really quickly",
            "God, just shut up already",
            "They were the nation's biggest business",
            "No woman would buy that",
            "I like furry animals",
            "I just saw the most eccentric brick",
            "We need congressional support",
            "Nobody is out there trying to be happy",
            "When they first met, they didn\'t get along",
            "This will make a real difference",
            "In city after city, big public meetings were held",
            "This is my big break",
            "Her oven had a habit of setting off the fire alarm for no reason",
            "Many trees are in the forest",
            "I'm not really a TV watcher",
            "I think I could fall asleep really quickly",
            "God, just shut up already",
            "They were the nation's biggest business",
            "No woman would buy that",
            "I like furry animals",
            "The course starts next Sunday",
            "These guys are the best",
            "The teacher tested us in English",
            "There were elements of his singing voice that reminded her of JC from N-Sync",
            "Why can you open the door?"
        ]
        self.sentence_embeddings = self.create_embeddings()
        self.l2_index =  self.create_l2_index()

        print("Done initializing RAG Search")

    def create_embeddings(self):
        """Create the embeddings for data."""
        sentence_embeddings = []
        for text in self.data: 
            sentence_embeddings.append(single_embedding(text))
        return np.array(sentence_embeddings)
        
    def create_l2_index(self):
        import faiss
        l2_index = faiss.IndexFlatL2(EMBEDDING_DIM)
        l2_index.add(self.sentence_embeddings)
        return l2_index
    
    # Brute force L2 Search
    def L2_search(self, query, top_k):
        query_emb = single_embedding(query).reshape(-1, EMBEDDING_DIM)
        return self.l2_index.search(query_emb, top_k)
    
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "Learning about Artificial Intelligence"
    print(rag_search.L2_search(query, 2))



