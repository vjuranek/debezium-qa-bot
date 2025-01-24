#!/usr/bin/env python3

from langchain_community.document_loaders import WebBaseLoader
from langchain_milvus import Milvus
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


CHUNK_SIZE = 250
CHUNK_OVERLAP = 0
EMBEDDINGS_MODEL = "llama3.2"
URLS = [
    "https://debezium.io/documentation/reference/3.0/tutorial.html",
    # "https://debezium.io/documentation/reference/3.0/configuration/avro.html",
    # "https://debezium.io/documentation/reference/3.0/configuration/topic-auto-create-config.html",
    # "https://debezium.io/documentation/reference/3.0/configuration/signalling.html",
    # "https://debezium.io/documentation/reference/3.0/configuration/notification.html",
    # "https://debezium.io/documentation/reference/3.0/connectors/oracle.html",
]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
embeddings = OllamaEmbeddings(model=EMBEDDINGS_MODEL)

docs = [WebBaseLoader(url).load() for url in URLS]
doc_splits = text_splitter.split_documents(docs[0])

Milvus.from_documents(
    documents=doc_splits,
    embedding=embeddings,
    connection_args={
        "host": "127.0.0.1:19530",
    },
    drop_old=True,
)
