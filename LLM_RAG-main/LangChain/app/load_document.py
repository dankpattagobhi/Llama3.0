from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
import os 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
import time 

class FileLoader:
    def __init__(self, directory_path=None, files_path=None, allowed_extensions=None):
        if allowed_extensions is None:
            allowed_extensions = ['.docx', 'pdf', 'txt'] 
        self.allowed_extensions = allowed_extensions
        self.directory_path = directory_path
        self.files_path = files_path

    def list_files(self):
        files_list = []
        if self.files_path == None:
            for root, dirs, files in os.walk(self.directory_path):
                for file in files:                    
                    if any(file.endswith(ext) for ext in self.allowed_extensions):
                        file_path = os.path.join(str(root), str(file))
                        files_list.append(file_path)
        else:
            for file in self.files_path:
                files_list.append(file)
        
        return files_list

    def load_files(self):
        documents = []
        for root, dirs, files in os.walk(self.directory_path):
            for file in files:
                if any(file.endswith(ext) for ext in self.allowed_extensions):
                    file_path = os.path.join(root, file)
                    print(file)
                    if file.endswith('.docx'):
                        loader = Docx2txtLoader(file_path)
                        documents.extend(loader.load())
                    elif file.endswith('.txt'):
                        with open(file_path, 'r') as f:
                            documents.append(f.read())
                    elif file.endswith('.pdf'):
                        loader = PyPDFLoader(file_path)
                        documents.extend(loader.load())
        return documents

    def load_selected_files(self):
        documents = []
        for file_path in self.files_path:
            if any(file_path.endswith(ext) for ext in self.allowed_extensions):
                if file_path.endswith('.docx'):
                    loader = Docx2txtLoader(file_path)
                    documents.extend(loader.load())
                elif file_path.endswith('.txt'):
                    with open(file_path, 'r') as f:
                        documents.append(f.read())
                elif file_path.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                    documents.extend(loader.load())
        return documents

    def text_splitting(self, documents, chunk_size=500, chunk_overlap=0):
        print("Starting text splitting...")
        start_time = time.time()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,  # Size of each chunk
            chunk_overlap=chunk_overlap  # Overlap between chunks for better context preservation
        )
        
        all_splits = text_splitter.split_documents(documents)  # Split into chunks

        end_time = time.time()
        
        print(f"Text splitting complete. Number of splits: {len(all_splits)}")
        print(f"Text splitting took {end_time - start_time:.2f} seconds.")

        return all_splits