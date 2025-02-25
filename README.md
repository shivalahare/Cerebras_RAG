
# Retrieval-Augmented Generation (RAG) System

This project implements a **Retrieval-Augmented Generation (RAG)** system that leverages **FAISS** for fast vector search, **Sentence-BERT** for document embeddings, and **Cerebras API** for natural language question answering. The goal of this project is to dynamically retrieve relevant content from documents and provide context-based answers to user queries.

## Features

- **Document Processing**: Extracts text from PDF and image files (using OCR for images).
- **Vector Search**: Uses **FAISS** for efficient document retrieval based on semantic similarity.
- **Question Answering**: Leverages **Cerebras API** for generating answers using the most relevant document as context.
- **OCR for Scanned Documents**: Supports image-based text extraction using **Tesseract OCR**.

## Requirements

To run the project, you need to have the following dependencies installed:

- Python 3.8+
- **FAISS** (for vector search)
- **Sentence-Transformers** (for embeddings)
- **Cerebras SDK** (for using the Cerebras API)
- **pdfplumber** (for extracting text from PDFs)
- **Pillow** (for image processing)
- **pytesseract** (for OCR)

You can install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```

### Dependencies:

- **faiss-cpu** or **faiss-gpu** (depending on your hardware)
- **sentence-transformers**
- **pdfplumber**
- **pytesseract**
- **Pillow**
- **cerebras**

## Setup

1. **Cerebras API Key**:
   - Sign up at [Cerebras](https://www.cerebras.net/) and obtain an API key.
   - Create a `.env` file in the root directory of the project and add the following:

   ```env
   CEREBRAS_API_KEY=your_api_key_here
   ```

2. **Tesseract OCR Setup**:
   - Install Tesseract OCR if you don't have it installed:
     - **Windows**: Download from [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki).
     - **Linux**: Install using your package manager (e.g., `sudo apt install tesseract-ocr`).

3. **Run the Application**:
   - Once you have the dependencies installed and your environment set up, run the script:

   ```bash
   python main.py
   ```

## Usage

1. The script processes documents, including PDFs and images. You can specify the file paths in the `input_files` list in the `main.py` script.
2. After processing the documents, the system will prompt you to ask questions.
3. The system will search for the most relevant document and use it to generate an answer through the **Cerebras API**.
4. To stop the process, type `exit`.

### Example Output

```
Processing file: Testing_Document.pdf

Processing file: Cerebras Python API library.pdf

All documents processed successfully!

Your Question (type 'exit' to quit): What is NLP?

Retrieving the most relevant document...

Fetching the answer...

Answer: According to Section 2: Overview of AI Technologies, under the subheading "Natural Language Processing (NLP)", it is stated that: "Natural Language Processing (NLP): Enables machines to understand, interpret, and generate human language."
```

## How it Works

1. **Document Reading**:
   - The system reads PDF files and images (via OCR). Text is extracted and stored for further processing.
   
2. **Vectorization**:
   - Each document is passed through **Sentence-BERT** to generate embeddings.
   - These embeddings are stored in a **FAISS** index for efficient similarity search.

3. **Query Handling**:
   - The user submits a query, which is also transformed into an embedding.
   - The **FAISS** index is queried to find the most similar document.
   
4. **Answer Generation**:
   - The relevant document is sent to **Cerebras API** along with the user query to generate a context-based answer.

## Contributing

Feel free to fork this repository and submit pull requests. Any contributions are welcome, especially if you have improvements for document processing, embedding techniques, or integration with other APIs.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
