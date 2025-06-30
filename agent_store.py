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

class AgentStore:
    def __init__(self,encoder, tokenizer):
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.agentlist = []
        self.registry = {}
    def add_agent(self, docstring, agent_name):
        inputs = self.tokenizer(docstring,return_tensors = 'pt').to(self.encoder.device)
        embedding = self.encoder(**inputs)['last_hidden_state'][:,0][0]
        embedding = embedding.detach().cpu()
        self.registry[len(self.agentlist)] = agent_name
        self.agentlist.append(embedding)
    def get_agent(self, query):
        inputs = self.tokenizer(query,return_tensors = 'pt').to(self.encoder.device)
        embedding = self.encoder(**inputs)['last_hidden_state'][:,0][0]
        embedding = embedding.detach().cpu()
        dot_products = [(i, F.cosine_similarity(embedding, emb, dim=0).item()) for i, emb in enumerate(self.agentlist)]
        max_idx, max_dot_product = max(dot_products, key=lambda x: x[1])
        return self.registry[max_idx]