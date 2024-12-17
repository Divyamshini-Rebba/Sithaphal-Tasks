import PyPDF2
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from bs4 import BeautifulSoup
from transformers import pipeline


class DataIngestion:
    def __init__(self, embedding_model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(embedding_model_name)
        self.vector_db = None
        self.metadata = []
        self.embeddings = []

    def crawl_and_scrape(self, urls):
        """Scrape data from the provided URLs and process the content."""
        for url in urls:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            content = soup.get_text()
            self.extract_data(content)

    def extract_data(self, content):
        """Extract text data and chunk it into manageable pieces."""
        chunks = content.split('\n\n')  # Split by paragraphs (could use more sophisticated methods)
        for chunk in chunks:
            if chunk.strip():  # Only process non-empty chunks
                self.process_chunk(chunk.strip())

    def process_chunk(self, chunk):
        """Process each chunk by creating embeddings and storing them."""
        embedding = self.model.encode(chunk)
        self.embeddings.append(embedding)
        self.metadata.append(chunk)

    def store_embeddings(self):
        """Store the embeddings in a Faiss vector database."""
        if not self.embeddings:
            raise ValueError("No embeddings to store.")
        dimension = len(self.embeddings[0])
        self.vector_db = faiss.IndexFlatL2(dimension)
        self.vector_db.add(np.array(self.embeddings).astype('float32'))
        return self.metadata


class QueryHandler:
    def __init__(self, ingestion_module):
        self.ingestion_module = ingestion_module

    def handle_query(self, user_query, k=5):
        """Handle the query by retrieving relevant chunks."""
        query_embedding = self.ingestion_module.model.encode(user_query)
        D, I = self.ingestion_module.vector_db.search(np.array([query_embedding]).astype('float32'), k)
        # Retrieve top relevant chunks
        relevant_chunks = [self.ingestion_module.metadata[i] for i in I[0]]
        return relevant_chunks


class ResponseGenerator:
    def __init__(self, model_name='gpt-2'):
        self.llm = pipeline('text-generation', model=model_name)

    def generate_response(self, relevant_chunks, user_query):
        """Generate a response using the relevant chunks and user query."""
        context = ' '.join(relevant_chunks)
        prompt = f"Based on the following information: {context}, answer the question: {user_query}"
        response = self.llm(prompt, max_length=100)[0]['generated_text']
        return response


class RAGPipeline:
    def __init__(self, pdf_url=None, urls=None, user_query=None):
        self.pdf_url = pdf_url
        self.urls = urls
        self.user_query = user_query
        self.ingestion = DataIngestion()
        self.query_handler = QueryHandler(self.ingestion)
        self.response_generator = ResponseGenerator()

    def extract_text_from_pdf(self, pdf_url):
        """Extract text from PDF URL."""
        response = requests.get(pdf_url)
        with open('temp.pdf', 'wb') as f:
            f.write(response.content)
        
        text = ""
        with open('temp.pdf', 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text()
        return text

    def run(self):
        """Run the entire RAG pipeline."""
        # If a PDF URL is provided, process the PDF
        if self.pdf_url:
            text = self.extract_text_from_pdf(self.pdf_url)
            self.ingestion.extract_data(text)
        # Otherwise, crawl and scrape the provided URLs
        elif self.urls:
            self.ingestion.crawl_and_scrape(self.urls)

        # Store embeddings
        self.ingestion.store_embeddings()

        # Query handling
        relevant_chunks = self.query_handler.handle_query(self.user_query)
        response = self.response_generator.generate_response(relevant_chunks, self.user_query)
        
        return response


# Example Usage
pdf_url = "https://www.hunter.cuny.edu/dolciani/pdf_files/workshop-materials/mmcpresentations/tables-charts-and-graphs-with-examples-from.pdf"
user_query = "What is the unemployment information based on type of degree?"

# Create RAG pipeline instance and run
rag_pipeline = RAGPipeline(pdf_url=pdf_url, user_query=user_query)
response = rag_pipeline.run()
print(response)
