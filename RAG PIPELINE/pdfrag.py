import PyPDF2
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


class TextExtractor:
    def __init__(self, pdf_url):
        self.pdf_url = pdf_url

    def extract_text_from_pdf(self):
        """Extract text from the given PDF URL."""
        response = requests.get(self.pdf_url)
        with open('temp.pdf', 'wb') as f:
            f.write(response.content)
        
        text = ""
        with open('temp.pdf', 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text()
        return text


class TextChunker:
    @staticmethod
    def chunk_text(text, chunk_size=512):
        """Chunk the extracted text into manageable pieces."""
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


class EmbeddingManager:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def create_embeddings(self, chunks):
        """Create embeddings for each chunk of text."""
        return self.model.encode(chunks)

    def store_embeddings(self, embeddings):
        """Store embeddings in a Faiss index."""
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        return index


class QueryHandler:
    def __init__(self, index, model):
        self.index = index
        self.model = model

    def query_embeddings(self, query, k=5):
        """Query the Faiss index to find the most relevant chunks."""
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(np.array([query_embedding]).astype('float32'), k)
        return indices[0]


class ResponseGenerator:
    def generate_response(self, retrieved_chunks, user_query):
        """Generate a response based on the retrieved chunks and user query."""
        context = ' '.join(retrieved_chunks)
        return f"Response based on: {context} for query: {user_query}"


class RAGPipeline:
    def __init__(self, pdf_url, user_query):
        self.pdf_url = pdf_url
        self.user_query = user_query
        self.text_extractor = TextExtractor(pdf_url)
        self.chunker = TextChunker()
        self.embedding_manager = EmbeddingManager()
        self.response_generator = ResponseGenerator()

    def run(self):
        """Run the entire RAG pipeline."""
        # Extract text from the PDF
        text = self.text_extractor.extract_text_from_pdf()
        
        # Chunk the extracted text
        chunks = self.chunker.chunk_text(text)

        # Create embeddings for the chunks
        embeddings = self.embedding_manager.create_embeddings(chunks)

        # Store embeddings in a Faiss index
        index = self.embedding_manager.store_embeddings(embeddings)

        # Query the index for the most relevant chunks
        query_handler = QueryHandler(index, self.embedding_manager.model)
        retrieved_indices = query_handler.query_embeddings(self.user_query)

        # Retrieve the relevant chunks from the index
        retrieved_chunks = [chunks[i] for i in retrieved_indices]

        # Generate a response based on the relevant chunks
        response = self.response_generator.generate_response(retrieved_chunks, self.user_query)
        return response


# Example usage
pdf_url = "https://www.hunter.cuny.edu/dolciani/pdf_files/workshop-materials/mmcpresentations/tables-charts-and-graphs-with-examples-from.pdf"
user_query = "What is the unemployment information based on type of degree?"

# Create and run the RAG pipeline
rag_pipeline = RAGPipeline(pdf_url=pdf_url, user_query=user_query)
response = rag_pipeline.run()
print(response)
