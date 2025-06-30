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

class Agent:
    def __init__(self, name: str, description: str, func: Callable):
        self.name = name
        self.description = description

    def generate():
        pass 

class CodeGenerator(Agent):
    def __init__(self,model, tokenizer, prompt_template, correction_prompt_template, pattern_correction_prompt_template):
        self.model = model
        self.tokenizer = tokenizer
        # self.arag_obj = arag_obj
        self.prompt_template = prompt_template
        self.correction_prompt_template = correction_prompt_template
        self.pattern_correction_prompt_template = pattern_correction_prompt_template 
        
    def generate(self,prompt):
        full_prompt = self.prompt_template(prompt)
        pattern = r"```python\s+([\s\S]+?)\s+```"
        response, flag, po = generate_until_pattern(self.model, self.tokenizer, full_prompt,pattern)
        response = response[len(full_prompt):].strip()
        if flag:
            try:
                match = re.findall(pattern, response)
                exec(match[0])
                return match[0]
            except:
                full_prompt = self.correction_prompt_template(response)
                response, _ , _ = generate_until_pattern(self.model, self.tokenizer,full_prompt, pattern)
                match = re.findall(pattern, response)
                if(len(match)==0):
                    return response
                return match[0]
        else:
            full_prompt = self.pattern_correction_prompt_template(response)
            respone, _ , _ = generate_until_pattern(self.model, self.tokenizer,full_prompt, pattern)
            match = re.findall(pattern, response)
            if(len(match)==0):
                return response
            return response
        
class DocumentationGenerator(Agent):
    def __init__(self, model, tokenizer, prompt_template, pattern_correction_prompt_template):
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        self.pattern_correction_prompt_template = pattern_correction_prompt_template
    def generate(self, prompt):
        full_prompt = self.prompt_template(prompt)
        pattern = r'```json\s*({[\s\S]*?"name"\s*:\s*"[^"]*"[\s\S]*?"description"\s*:\s*"[^"]*"[\s\S]*?"parameters"\s*:\s*{[\s\S]*?}[\s\S]*?"returns"\s*:\s*"[^"]*"\s*})\s*```'
        response, flag, po = generate_until_pattern(self.model, self.tokenizer, full_prompt,pattern)
        return response
    
class HRManager(Agent):
    def __init__(self,model,tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    def generate(self,prompt):
        inputs = self.tokenizer(prompt,return_tensors = 'pt').to(self.model.device)
        outputs = model.generate(
            **inputs,
            pad_token_id = self.tokenizer.eos_token_id,
            do_sample = False,
            max_new_tokens = 2048
        )
        return self.tokenizer.decode(outputs[0],skip_special_tokens = True)[len(prompt):].strip()

class MathExpert(Agent):
    def __init__(self,model,tokenizer, prompt_template):
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
    def generate(self,prompt):
        prompt = self.prompt_template(prompt)
        inputs = self.tokenizer(prompt,return_tensors = 'pt').to(self.model.device)
        outputs = model.generate(
            **inputs,
            pad_token_id = self.tokenizer.eos_token_id,
            do_sample = False,
            max_new_tokens = 2048
        )
        return self.tokenizer.decode(outputs[0],skip_special_tokens = True)[len(prompt):].strip()

