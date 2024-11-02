# **DocRAG with ChromaDB and FAISS**

This project implements a **Retrieval-Augmented Generation (RAG) Query Application** that integrates FAISS for efficient vector search, Ollama’s Llama 2 model to generate context-aware responses to user queries and ChromaDB for persistent storage. The pipeline is designed to process research papers and provides AI-driven, accurate answers by combining advanced retrieval and generation techniques.

## Project Highlights
- **Efficient Vector Search with FAISS**: Retrieves relevant document chunks at high speed.
- **Contextual AI Responses**: Uses Ollama's Llama 2 model to generate insightful, context-specific answers.
---

## Table of Contents
1. [Installation](#installation)
2. [Project Structure](#project-structure)
3. [Usage](#usage)
4. [Results](#results)
5. [Furure work](#Future-work)
6. [License](#license)

---

## Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/EMoetez/DocRAG-with-ChromaDB.git
cd DocRAG-with-ChromaDB
```
### Step 2: Install Dependencies
Install all required dependencies:
```bash
pip install -r requirements.txt
```
Project Structure

project_name/
│
├── notebooks/                # Directory for Jupyter notebooks (.ipynb)
│   └── DocRAGChromaDB.ipynb  # Main notebook
│
├── src/                      # Directory for source code (.py files)
│   └── main_script.py        # Script for RAG pipeline
│
├── requirements.txt          # List of dependencies
├── README.md                 # Project documentation
└── .gitignore                # Files/directories to ignore in Git
## Usage
The project is organized for easy setup and usage. Follow these steps to use the RAG Query Application.

### 1. Set Up Your Vector Store and Database
ChromaDB and FAISS are used to store and search vectorized document chunks.

### 2. Load Documents and Split Text
Load documents, such as research papers, and split them into chunks for optimized retrieval and response generation.

### 3. Embed and Store Chunks in ChromaDB and FAISS
Embed the document chunks for vector representation and store these embeddings in ChromaDB and FAISS.

### 4. Query the Model
With the vectors stored, you can now query the RAG pipeline. FAISS retrieves relevant documents, and Ollama’s Llama 2 generates context-specific responses.

## Results

This image below demonstrates a typical interaction, showcasing a user query and the AI-generated response.
![Screenshot_20241102_094631](https://github.com/user-attachments/assets/e4e5558d-c495-43ca-841a-df8044f297dc)

## Future work
In the next steps, we will use ChromaDB for more effecient storage. ChromaDB Stores document embeddings for future retrieval.

## License
This project is licensed under the MIT License.




