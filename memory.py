from variables import tavily_api, auth_token
from huggingface_hub import login
login(token=auth_token)
import openmeteo_requests
import torch
import torch.nn.functional as F
import json
import re
from datasets import load_dataset
import requests
from datetime import datetime
import requests_cache
import pandas as pd
import copy
from retry_requests import retry
from tqdm.notebook import tqdm
import time
import numpy as np
from typing import List, Dict, Tuple, Optional, Literal, Union, Callable, Any
from dataclasses import dataclass
from enum import Enum
from tavily import TavilyClient
from transformers import T5Tokenizer, T5ForSequenceClassification, T5ForConditionalGeneration
from transformers import PreTrainedTokenizer, PreTrainedModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import langchain
from ragclient import RAGClient
from typing import List
from sentence_transformers import SentenceTransformer
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Core_Memory:
    def __init__(self, embedder, device):
        self.device = device
        index = faiss.IndexFlatL2(len(embedder.embed_query("hello world")))
        self.vector_store = FAISS(
            embedding_function=embedder,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        self.memory = []
    def add_qna(self, q, a, docs):
        self.memory.append((q, a, docs))
        doc = Document(page_content=f"Past sub-query: {q}\nSub-query answer: {a}\nDocuments: {docs}",
                      metadata={'id':len(self.memory)})
        self.vector_store.add_documents(documents = [doc], #entities inside this list are pushed in vector store as seperate independent docs
                                       ids = [len(self.memory)])
    def get_relevant_memories(self, query, topk=1):
        results = self.vector_store.similarity_search(
            query,
            k=topk,
            filter={},
        )
        out = [res.page_content for res in results]
        return out

class ShortTermMemory:
    """
    Simple BGE-based Short-Term Memory Implementation
    """
    def __init__(self, cross_encoder=None, tokenizer=None, pathwayobj=None):
        """
        Initialize the Short-Term Memory system.
        
        Args:
            cross_encoder: Optional cross-encoder model
            tokenizer: Optional tokenizer
            pathwayobj: Optional pathway object
        """
        # Model for embedding documents and queries
        self.query_embed_model = SentenceTransformer('BAAI/bge-base-en-v1.5')
        
        # Optional additional components
        self.cross_encoder = cross_encoder
        self.tokenizer = tokenizer 
        self.pathwayobj = pathwayobj
        
        # Embedding instructions
        self.DEFAULT_EMBED_INSTRUCTION = "Represent the document for retrieval: "
        self.DEFAULT_QUERY_INSTRUCTION = (
            "Represent the question for retrieving supporting documents: "
        )
        self.DEFAULT_QUERY_BGE_INSTRUCTION_EN = (
            "Represent this question for searching relevant passages: "
        )
        
        # Storage for queries
        self._query_texts = []
        self._query_embeddings = []

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Compute document embeddings using a HuggingFace transformer model.
        
        Args:
            texts: The list of texts to embed.
        
        Returns:
            List of embeddings, one for each text.
        """
        # Prepare texts with embedding instruction
        prepared_texts = [
            f"{self.DEFAULT_EMBED_INSTRUCTION}{t.replace('\n', ' ')}" 
            for t in texts
        ]
        
        # Encode texts
        embeddings = self.query_embed_model.encode(prepared_texts)
        
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        Compute query embeddings using a HuggingFace transformer model.
        
        Args:
            text: The text to embed.
        
        Returns:
            Embeddings for the text.
        """
        # Prepare query with embedding instruction
        prepared_text = f"{self.DEFAULT_QUERY_INSTRUCTION}{text.replace('\n', ' ')}"
        
        # Encode query
        embedding = self.query_embed_model.encode(prepared_text)
        
        return embedding.tolist()

    def add_query_text(self, query_text: str):
        """
        Add query text to the short-term memory and compute its embedding.
        
        Args:
            query_text: The query text to add.
        """
        # Add query text
        self._query_texts.append(query_text)
        
        # Compute and store embedding
        query_embedding = self.embed_query(query_text)
        self._query_embeddings.append(query_embedding)

    def get_relevant_short_memory(self, query: str, top_k: int = 3):
        """
        Retrieve most relevant queries from short-term memory.
        
        Args:
            query: The query to find relevant memories for.
            top_k: Number of top relevant memories to return.
        
        Returns:
            List of most relevant query texts.
        """
        # If no memories exist, return empty list
        if not self._query_texts:
            return []
        
        # Embed the input query
        query_embedding = self.embed_query(query)
        
        # Compute cosine similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(
            [query_embedding], 
            self._query_embeddings
        )[0]
        
        # Get indices of top-k most similar memories
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        # Return corresponding query texts
        return [self._query_texts[idx] for idx in top_indices]

