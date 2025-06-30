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

class CorrectiveRAGFilter:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        upper_threshold: float = 0.5,
        lower_threshold: float = -0.9,
        filter_threshold: float = -0.5,
        top_n: int = 5,
        max_length: int = 512,
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizer
        self.model = model.to(self.device)
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.filter_threshold = filter_threshold
        self.max_length = max_length
        self.top_n = top_n
    
    def split_into_passages(self, psg: str, mode: str = "excerption") -> List[str]:
        if mode == 'fixed_num':
            final_strips = []
            window_length = 50
            words = psg.split(' ')
            buf = []
            for w in words:
                buf.append(w)
                if len(buf) == window_length:
                    final_strips.append(' '.join(buf))
                    buf = []
            if buf != []:
                if len(buf) < 10:
                    final_strips[-1] += (' ' + ' '.join(buf))
                else:
                    final_strips.append(' '.join(buf))
            return final_strips
        
        if mode == 'excerption':
            num_concatenate_strips = 3
            question_strips = psg.split('?')
            origin_strips = []
            for qs in question_strips:
                origin_strips += qs.split('. ')
            strips = []
            for s in origin_strips:
                if s in strips:
                    continue
                if strips == []:
                    strips.append(s)
                else:
                    if len(s.split()) > 5:
                        strips.append(s)
                    else:
                        strips[-1] += s
            final_strips = []
            buf = []
            for strip in strips:
                buf.append(strip)
                if len(buf) == num_concatenate_strips:
                    final_strips.append(' '.join(buf))
                    buf = []
            if buf != []:
                final_strips.append(' '.join(buf))
            return final_strips
        elif mode == 'selection':
            return [psg]
        
    def get_relevant_strips(self, strips: List[str], query: str) -> str:
        strips_data = []
        for p in strips:
            if len(p.split()) < 4:
                scores = -1.0
            else:
                input_content = query + " [SEP] " + p
                inputs = self.tokenizer(input_content, return_tensors = "pt", padding = "max_length", truncation = True, max_length = self.max_length)
                try:
                    with torch.no_grad():  
                        outputs = self.model(inputs["input_ids"].to(self.device), 
                                        attention_mask=inputs["attention_mask"].to(self.device))
                    scores = float(outputs["logits"].cpu())
                except:
                    scores = -1.0
            strips_data.append((scores, p))
        
        def take_idx(elem):
            return elem[0]
        sorted_results = sorted(strips_data, key = take_idx, reverse = True)[:]
        filtered_results = [data for data in sorted_results if data[0] > self.filter_threshold]
        ctxs = [s[1] for s in filtered_results[:self.top_n]]
        
        return '; '.join(ctxs)
    
    def score_and_relevance_flag(self, query: str, docs: List[str]) -> int:
        self.model.eval()
        
        scores = []
        for doc in docs:
            input_content = query + " [SEP] " + doc
            inputs = self.tokenizer(input_content, return_tensors = "pt", padding = "max_length", truncation = True, max_length = self.max_length)
            try:
                with torch.no_grad():  
                    outputs = self.model(inputs["input_ids"].to(self.device), 
                                    attention_mask=inputs["attention_mask"].to(self.device))
                scores.append(float(outputs["logits"].cpu()))
            except:
                scores.append(-1.0)
            
        incorrect_flag = 0
        
        for score in scores:
            if score >= self.upper_threshold:
                return 2
            elif score < self.lower_threshold:
                incorrect_flag += 1
            
        if incorrect_flag == len(docs):
            return 0
        
        return 1
    
    def web_search_and_filter(self, query: str) -> str:
        tavily_client = TavilyClient(api_key = tavily_api)
        response = tavily_client.search(query, max_results = 5)
        contents = []
        for i in range(5):
            content = response['results'][i]['content']
            contents += self.split_into_passages(content)
        final_strips = self.get_relevant_strips(contents, query)
        return final_strips
    
    def C_RAG(self, query: str, docs: List[str]) -> str:
        flag = self.score_and_relevance_flag(query, docs)
        if flag == 2:
            internal_info = []
            for p in docs:
                internal_info += self.split_into_passages(p)

            return self.get_relevant_strips(internal_info, query)
        
        elif flag == 0:
            return self.web_search_and_filter(query)
        
        else:
            internal_info = []
            for p in docs:
                internal_info += self.split_into_passages(p)
                
            external_info = self.web_search_and_filter(query)
            
            return self.get_relevant_strips(internal_info, query) + "; " + external_info

