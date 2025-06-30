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

class QueryDecomposer:
    def __init__(self, model, tokenizer, prompt_template) -> List[str]:
        self.tokenizer = tokenizer
        self.model = model
        self.prompt_template = prompt_template
        
    def generate_sub_queries(self, prompt: str, max_new_tokens: int = 2048) -> List:
        full_prompt = self.prompt_template(prompt)
        pattern = r'\{\s*"single_hop_queries"\s*:\s*\[\s*(?:"[^"]*"\s*,\s*)*"[^"]*"\s*\]\s*\}'
        response, flag, po = generate_until_pattern(self.model, self.tokenizer, full_prompt,
                             pattern)
        response = response[len(full_prompt):].strip()
        print(response)
        return json.loads(re.findall(pattern, response)[0])['single_hop_queries'] #maybe .replace