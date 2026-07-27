import os
from dotenv import load_dotenv
#from langchain_community.document_loaders import Textloader #deprecated
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if __name__ == '__main__':
    print("Ingesting...")
    loader = UnstructuredLoader(file_path=r"C:\Users\ebere\langchain-course\mediumblog1.txt", chunking_strategy="basic", max_characters=1000000)#initialize text loader
    #loader = UnstructuredLoader(file_path="C:\\Users\\ebere\\langchain-course\\mediumblog1.txt", chunking_strategy="basic", max_characters=1000000)#initialize text loader or just use / for filepath
    document = loader.load() #method to load file into langchain document

    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)#each chunk has 1k xters long and no overlap between chunks, don't repeat text
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    print("ingesting...")
    PineconeVectorStore.from_documents(
         texts, embeddings, index_name=os.environ["INDEX_NAME"]
    )
    print("finish")
    #print(os.environ['PINECONE_API_KEY'])
