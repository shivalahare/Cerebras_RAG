import os
import pdfplumber
import faiss
import numpy as np
from dotenv import load_dotenv
from cerebras.cloud.sdk import Cerebras
from PIL import Image
import pytesseract
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()
API_KEY = os.getenv("CEREBRAS_API_KEY")

# Initialize Cerebras Client
client = Cerebras(api_key=API_KEY)

# Initialize Sentence-BERT model for embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize FAISS index for document retrieval
dimension = 384  # Dimensionality of Sentence-BERT embeddings
index = faiss.IndexFlatL2(dimension)

# PDF Reading Function
def read_pdf(file_path):
    data = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                data += page.extract_text()
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return data

# OCR for Images or Scanned PDFs
def extract_text_with_ocr(image_path):
    try:
        return pytesseract.image_to_string(Image.open(image_path))
    except Exception as e:
        print(f"Error with OCR: {e}")
        return ""

# Add document content to FAISS index
def add_documents_to_index(documents):
    for doc in documents:
        embeddings = embedder.encode([doc])
        faiss.normalize_L2(embeddings)
        index.add(np.array(embeddings, dtype=np.float32))

# Retrieve the most relevant document based on query
def retrieve_relevant_documents(query):
    query_embedding = embedder.encode([query])
    faiss.normalize_L2(query_embedding)
    
    # Search for the most similar document
    _, I = index.search(np.array(query_embedding, dtype=np.float32), k=1)
    
    return documents[I[0][0]] if I[0][0] != -1 else None

# Get QA Solution from Cerebras
def generate_rag_response(user_query, context):
    try:
        # Sending the request to Cerebras API
        print("\nSending request to Cerebras...")
        response = client.chat.completions.create(
            messages=[ 
                {"role": "system", "content": "You are a helpful assistant for document analysis."},
                {"role": "user", "content": f"Document content: {context}"},
                {"role": "user", "content": f"Question: {user_query}"}
            ],
            model="llama3.3-70b"  # Change the model if required
        )

        # Check if the response contains choices and extract the message content
        if response.choices and len(response.choices) > 0:
            choice = response.choices[0]
            return choice.message.content if hasattr(choice.message, 'content') else None
        else:
            print("Unexpected response structure:", response)
            return None
    except Exception as e:
        print(f"Error generating response: {e}")
        return None

# Main Function with Combined Context and Dynamic Retrieval
def main():
    # Input Documents (Different File Types)
    input_files = [
        "Testing_Document.pdf",
        "Cerebras Python API library.pdf",
    ]

    global documents
    documents = []  # List to store extracted documents text

    # Loop through each file and extract content
    for input_file in input_files:
        print(f"\nProcessing file: {input_file}")

        content = ""
        if input_file.endswith(".pdf"):
            content = read_pdf(input_file)  # Extract text from PDF
        elif input_file.endswith(('.png', '.jpg', '.jpeg')): 
            content = extract_text_with_ocr(input_file)  # Extract text from image
        else:
            print(f"Unsupported file format: {input_file}")
            continue  # Skip unsupported files

        if content.strip():  # Add only non-empty content
            documents.append(content)

    if not documents:
        print("No valid content found in the input files.")
        return

    print("\nAll documents processed successfully!")

    # Add documents to FAISS index
    add_documents_to_index(documents)

    # Start QA session with dynamic retrieval of context
    while True:
        question = input("\nYour Question (type 'exit' to quit): ")
        if question.lower() == "exit":
            print("Exiting the QA Agent. Goodbye!")
            break

        print("\nRetrieving the most relevant document...")
        relevant_document = retrieve_relevant_documents(question)

        if relevant_document:
            print("\nFetching the answer...")
            answer = generate_rag_response(question, relevant_document)
            if answer:
                print(f"Answer: {answer}")
            else:
                print("Failed to fetch the answer. Please try again.")
        else:
            print("No relevant documents found for the query.")

if __name__ == "__main__":
    main()
