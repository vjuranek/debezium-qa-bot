#!/usr/bin/env python3

import logging

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

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format=("%(levelname)-5s [%(name)s] %(message)s"))
log = logging.getLogger("loader")

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
embeddings = OllamaEmbeddings(model=EMBEDDINGS_MODEL)

def load_from_url(url):
    log.info(f"Processing URL {url!r}")
    docs = WebBaseLoader(url).load()
    doc_split = text_splitter.split_documents([doc for doc in docs])

    log.info("Loading splited document into the DB")
    Milvus.from_documents(
        documents=doc_split,
        embedding=embeddings,
        connection_args={
            "host": "127.0.0.1:19530",
        },
        drop_old=True,
    )


if __name__ == "__main__":
    load_from_url(URLS[0])
    
