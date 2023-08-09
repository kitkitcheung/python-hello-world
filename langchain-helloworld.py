from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings

import os

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
all_splits = text_splitter.split_documents(data)

import os
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_BASE"] = os.getenv("AZURE_OPENAI_ENDPOINT") 
os.environ["OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_KEY")
os.environ["OPENAI_API_VERSION"] = "2023-05-15"
question = "What are the approaches to Task Decomposition?"

embeddings=OpenAIEmbeddings(
    deployment="text-embedding-ada-002",
    model="text-embedding-ada-002",
    chunk_size=16)

print("Example using Chroma as vectorstore")
from langchain.vectorstores import Chroma
vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings)
docs = vectorstore.similarity_search(question)
print(docs[0].page_content)

print("Example using FAISS as vectorstore")
from langchain.vectorstores import FAISS
vectorstore = FAISS.from_documents(documents=all_splits, embedding=embeddings)
docs = vectorstore.similarity_search(question)
print(docs[0].page_content)

print("Example using LanceDB as vectorstore")
from langchain.vectorstores import LanceDB
import lancedb

db = lancedb.connect("./lancedb")
table = db.create_table(
    "my_table",
    data=[
        {
            "vector": embeddings.embed_query("Hello World"),
            "text": "Hello World",
            "id": "1",
        }
    ],
    mode="overwrite",
)
docsearch = LanceDB.from_documents(all_splits, embeddings, connection=table)

docs = docsearch.similarity_search(question)

print(docs[0].page_content)

# from langchain.retrievers import SVMRetriever
# svm_retriever = SVMRetriever.from_documents(all_splits,OpenAIEmbeddings(
#     deployment="text-embedding-ada-002",
#     model="text-embedding-ada-002",
#     chunk_size=1))
# docs_svm=svm_retriever.get_relevant_documents(question)

# print(len(docs_svm))