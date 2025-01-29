import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

# Load the scraped data
data_path = "data/technical_courses.parquet"
textChunksDF = pd.read_parquet(data_path)

# Load the SentenceTransformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Define FAISS index file path
faiss_index_file = "embeddings/faiss_index.bin"

# Check if FAISS index exists, else create a new one
if os.path.exists(faiss_index_file):
    faiss_index = faiss.read_index(faiss_index_file)
    print("✅ Loaded existing FAISS index.")
else:
    # Create embeddings for the chunks
    chunks_embeddings = embedding_model.encode(textChunksDF['text'].tolist())

    # Initialize FAISS index
    dimension = chunks_embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)

    # Add embeddings to FAISS index
    faiss_index.add(np.array(chunks_embeddings).astype(np.float32))

    # Save FAISS index to disk
    faiss.write_index(faiss_index, faiss_index_file)
    print("✅ Created and saved FAISS index.")

# Function to find relevant chunk
def find_relevant_chunk(query):
    query_embedding = embedding_model.encode([query])
    k = 1
    _, indices = faiss_index.search(np.array(query_embedding).astype(np.float32), k)
    return textChunksDF['text'].iloc[indices[0][0]]