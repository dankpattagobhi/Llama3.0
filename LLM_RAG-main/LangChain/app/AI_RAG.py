import time
import os
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain import hub
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from app.load_document import FileLoader
import re
import shutil  # For deleting the vectorstore directory
import signal
import sys

# Set up the environment variable (if required)
os.environ["LANGCHAIN_API_KEY"] = ""

class RAG_LLAMA:
    def __init__(self, model="llama3:latest", rag_prompt="rlm/rag-prompt", dir_path="./uploads", files_path=None, previous_files_path=None, previous_dir_path=None, persist_directory=None):
        self.vectorstore = None
        self.qa_chain = None
        self.model = model
        self.rag_prompt = rag_prompt
        self.dir_path = dir_path
        self.persist_directory = persist_directory
        self.files_path = files_path
        self.previous_files_path = previous_files_path
        self.previous_dir_path = previous_dir_path
        self.embeddings = OllamaEmbeddings(model=self.model)
        self.document_embeddings = None
        self.document_texts = None

        signal.signal(signal.SIGINT, self.cleanup)


    def cleanup(self, *args):
        """
        Cleanup resources before exiting, such as deleting the vectorstore directory.
        Called explicitly during SIGINT (Ctrl + C).
        """
        if self.persist_directory:
            try:
                print(f"Attempting to delete vectorstore directory: {self.persist_directory}")
                shutil.rmtree(self.persist_directory)
                print("Vectorstore directory deleted successfully.")
            except FileNotFoundError:
                print("Vectorstore directory not found. No action needed.")
            except Exception as e:
                print(f"Error while deleting vectorstore directory: {e}")
        else:
            print("No persist_directory specified. Skipping vectorstore cleanup.")

        # Exit the program gracefully
        print("Exiting program...")
        sys.exit(0)

    def embedding_and_vector(self, splits):
        print("Starting embedding process...")
        start_time = time.time()

        # Using Ollama embeddings for the vector store        
        if self.persist_directory == None:
            self.persist_directory = "./vectorstore/db_all"

        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
        )
        
        end_time = time.time()
        print(f"Embedding completed in {end_time - start_time:.2f} seconds.")
        
        return vectorstore

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def qa_chain_model(self, vectorstore):
        print("Setting up QA chain with Ollama...")
        start_time = time.time()

        llm = OllamaLLM(model=self.model)  # Using local Ollama LLM
        retriever = vectorstore.as_retriever()

        # Load custom prompt for RAG setup
        rag_prompt = hub.pull(owner_repo_commit=self.rag_prompt)
        
        qa_chain = (
            {"context": retriever | self.format_docs, "question":
            RunnablePassthrough()}
            | rag_prompt
            | llm
            | StrOutputParser()
        )

        end_time = time.time()
        print(f"QA Chain setup complete in {end_time - start_time:.2f} seconds.")
        
        return qa_chain

    def train_rag(self):
        fileObj = FileLoader(directory_path=self.dir_path, files_path=self.files_path)

        documents = []

        if self.dir_path == None:
            documents = fileObj.load_selected_files()
        else:
            documents = fileObj.load_files()
        
        splits = fileObj.text_splitting(documents=documents, chunk_size=500, chunk_overlap=0)

        
        self.document_texts = [doc.page_content for doc in splits]
        
        self.document_embeddings = self.embeddings.embed_documents(self.document_texts)

        # Embedding the splits and storing them in a vector store
        self.vectorstore = self.embedding_and_vector(splits=splits)
        
        # Setting up the question-answering chain
        self.qa_chain = self.qa_chain_model(vectorstore=self.vectorstore)



    def response_query(self, question):
        print("Processing query...")
        # Step 1: Check if the query relates to the document (simple keyword check)
        if self.is_document_related(question):
            print("Query is related to the document. Using RAG system.")
            # Process using the RAG QA chain
            start_time = time.time()
            answer = self.qa_chain.invoke(question)
            end_time = time.time()
            print(f"Answer received from RAG system: {answer}")
            print(f"Query processing took {end_time - start_time:.2f} seconds.")
            return answer
        else:
            print("Query is general. Using Ollama for general knowledge.")
            # Process using Ollama for general knowledge
            return self.ask_general_ollama(question)




    def is_document_related(self, question, threshold=0.4):
        # Check for specific keywords or patterns in the query
        # Embed the question/query
        document_keywords = [
            r"\b(list|outline|summarize|explain|describe|details|points)\b",
            r"(this document|inside the document|about the document|related to the document)",
            r"(content|sections|key takeaways|main idea|overview)",
        ]

        # Check if the query explicitly refers to the document
        for keyword in document_keywords:
            if re.search(keyword, question, re.IGNORECASE):
                print("Query explicitly refers to the document. Marking as related.")
                return True

        question_embedding = self.embeddings.embed_documents([question])
                
        # Calculate cosine similarity between the query and each document chunk
        similarities = cosine_similarity(question_embedding, self.document_embeddings)
        
        # Get the maximum similarity score (to handle multiple document chunks)
        max_similarity = np.max(similarities)
        
        print(f"Max similarity score: {max_similarity}")
        
        # If the maximum similarity exceeds the threshold, consider the query related to the document
        if max_similarity >= threshold:
            return True
        else:
            return False

    def ask_general_ollama(self, question):
        llm = OllamaLLM(model=self.model)
        start_time = time.time()
        answer = llm.invoke(question)
        end_time = time.time()
        print(f"Answer received from Ollama: {answer}")
        print(f"Query processing took {end_time - start_time:.2f} seconds.")
        return answer
    
    # Destructor
    def __del__(self):
        print("Destructor called.")
        self.cleanup()



if __name__ == "__main__":
    print("Initializing RAG-LLAMA...")
    start_time = time.time()

    # Initialize the RAG-LLAMA system with Ollama running locally
    obj = RAG_LLAMA(dir_path='./uploads/hr')
    
    end_time = time.time()
    print(f"RAG-LLAMA initialized in {end_time - start_time:.2f} seconds.")
    # Example: Respond to a query after initializing
    query = "What is the main idea of the document?"
    response = obj.response_query(obj.qa_chain, query)
    print(f"Response to query: {response}")