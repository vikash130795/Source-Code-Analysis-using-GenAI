import os
from git import Repo
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import Language
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings

# def create_and_get_folder_path(folder_name):
#   current_directory = os.getcwd()
#   global new_folder_path
#   new_folder_path = os.path.join(current_directory, folder_name)
#   os.makedirs(new_folder_path, exist_ok=True)
#   return new_folder_path

# global repo_folder_path

#clone any github repositories 
def repo_ingestion(repo_url):
    os.makedirs("repo", exist_ok=True)
    # global repo_folder_path
    # repo_folder_path = create_and_get_folder_path("repo")
    # #print(repo_folder_path) 
    repo_path = "repo/"
    Repo.clone_from(repo_url, to_path=repo_path)



#Loading repositories as documents
def load_repo(repo_path):
    # dir = os.path.abspath(os.getcwd())
    # path_dir = dir + "/repo"
    # print(path_dir)
    loader = GenericLoader.from_filesystem(repo_path,
                                        glob = "**/*",
                                       suffixes=[".py"],
                                       parser = LanguageParser(language=Language.PYTHON, parser_threshold=500)
                                        )
    
    documents = loader.load()

    return documents




#Creating text chunks 
def text_splitter(documents):
    documents_splitter = RecursiveCharacterTextSplitter.from_language(language = Language.PYTHON,
                                                             chunk_size = 2000,
                                                             chunk_overlap = 200)
    
    text_chunks = documents_splitter.split_documents(documents)

    return text_chunks



#loading embeddings model
def load_embedding():
    embeddings=OpenAIEmbeddings(disallowed_special=())
    return embeddings


