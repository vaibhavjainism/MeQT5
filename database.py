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

class Database:
    def __init__(self):
        self.rag_client = RAGClient(port=8080)
        self.JSON_DIR = "/workspace/codesearchnet/dataset.json"
        with open(self.JSON_DIR,'r') as f:
            self.code_book = json.load(f)
    def retrieve(self, query, top_k=1):
        retrieved_docs = [self.code_book[docstr] for docstr in self.rag_client.search_documents(query,top_k)]
        return retrieved_docs
class Database2:
    def __init__(self):
        self.rag_client = RAGClient(port=8081)
    def retrieve(self, query, top_k=1):
        return self.rag_client.search_documents(query,top_k)