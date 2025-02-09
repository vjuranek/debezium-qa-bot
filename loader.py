#!/usr/bin/env python3

import logging

from flask import Flask
from flask import request
from langchain_community.document_loaders import WebBaseLoader
from langchain_milvus import Milvus
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


CHUNK_SIZE = 250
CHUNK_OVERLAP = 0
EMBEDDINGS_MODEL = "llama3.2"

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format=("%(levelname)-5s [%(name)s] %(message)s"))
log = logging.getLogger("loader")

app = Flask("loader")

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
embeddings = OllamaEmbeddings(model=EMBEDDINGS_MODEL)

@app.route("/load")
def load_from_url():
    url = request.args.get("url")
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
    log.info(f"Loaded {url} into the DB")
    
    return f"Loaded {url}"


if __name__ == "__main__":
    app.run()
    
