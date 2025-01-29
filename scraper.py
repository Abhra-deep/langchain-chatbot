from langchain.document_loaders import WebBaseLoader
import pandas as pd

# Define the URL to scrape
url = "https://brainlox.com/courses/category/technical"

# Use Langchain WebBaseLoader to extract data
loader = WebBaseLoader(url)
documents = loader.load()

# Convert documents into DataFrame
data = pd.DataFrame([{"text": doc.page_content, "metadata": doc.metadata} for doc in documents])

# Save as Parquet for embeddings
data.to_parquet("data/technical_courses.parquet", index=False)

print("Data successfully scraped and saved!")
