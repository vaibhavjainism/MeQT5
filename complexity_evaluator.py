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

class A_RAG:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        LLM_model: PreTrainedModel,
        LLM_tokenizer: PreTrainedTokenizer,
        max_iters: int,
        maxlen: int,
        device: str,
        corrective_rag_object
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.LLM_model = LLM_model
        self.LLM_tokenizer = LLM_tokenizer
        self.max_iters = max_iters
        self.device = device
        self.max_length = maxlen
        self.converter = {'A': 0.0,
                         'B': 1.0,
                         'C': 2.0}
        self.corrector = corrective_rag_object
        
    def retrieve(self, query:str):
        return ["paris is capital of france"]
    def rerank(self,query: str, docs: List[str]):
        return docs
    def prompt_template_for_check(self, query: str, answer: str) -> str:
        return f'''Evaluate if the answer directly addresses the given question, regardless of factual accuracy:

Question: {query}
Answer: {answer}

Please analyze:
1. Does the answer attempt to respond to the specific question asked?
2. Are the main points of the question addressed in the answer?
3. Is the answer on-topic and relevant to the query?

Respond with:
- "YES" if the answer addresses the question
- "NO" if the answer is irrelevant or off-topic
- Brief explanation of why (1-2 sentences)
'''

    def get_prompt_for_no_retrieval(self, query: str):
        return f'''Provide a clear and comprehensive answer to the following question:

Question: {query}

Please follow these guidelines:
1. Give a direct, focused answer first
2. Provide relevant context and explanations if needed
3. Structure the response logically

If the question is unclear or needs clarification, please state what needs to be clarified before proceeding.'''

    def get_prompt_for_single_retrieval(self, query, docs):
        newline = '\n'
        context_sections = []
        for i, doc in enumerate(docs, 1):
            # Format each document with clear separation and reference number
            context_sections.append(f"Reference {i}:\n{doc}\n")
    
        # Construct a more detailed prompt with clear instructions
        prompt = f"""Question: {query}

Relevant Context:
{newline.join(context_sections)}

Please provide a comprehensive answer based on the context above. Include specific references to support your response when applicable."""

        return prompt

    def get_class(self, query: str):
        inputs = self.tokenizer(query, return_tensors = "pt", padding = "max_length", truncation = True, max_length = 512)
        # try:
        with torch.no_grad():  
            outputs = self.model.generate(inputs["input_ids"].to(self.device), 
                            attention_mask=inputs["attention_mask"].to(self.device), max_new_tokens = self.max_length)
        return self.converter[self.tokenizer.decode(outputs[0], skip_special_tokens = True)]
        # except:
        #     return float(1.0)

    def generate(self, prompt: str):
        inputs = self.LLM_tokenizer(prompt, return_tensors = "pt", padding = "max_length", truncation = True, max_length = 512).to(self.device)
        # try:
        with torch.no_grad():
            outputs = self.LLM_model.generate(**inputs, max_new_tokens = self.max_length)
        final_response = self.LLM_tokenizer.decode(outputs[0], skip_special_tokens = True)
        return final_response[len(prompt):].strip()
        # except:
        #     return prompt   
            
    def check(self, query, answer):
        flag = self.generate(self.prompt_template_for_check(query, answer))
        if "yes" in flag.lower():
            return True
        return False
        
    def no_retrieve(self, query: str):
        prompt = self.get_prompt_for_no_retrieval(query)
        return self.generate(prompt)

    def single_retrieve(self, query: str):
        docs = self.rerank(query, self.retrieve(query))
        docs = self.corrector.C_RAG(query, docs)
        prompt = self.get_prompt_for_single_retrieval(query, docs)
        return self.generate(prompt)
        
    def multi_retrieve(self, query: str):
        for _ in range(self.max_iters):
            docs = self.rerank(query, self.retrieve(query))
            docs = self.corrector.C_RAG(query, docs)
            updated_query = self.get_prompt_for_single_retrieval(query, docs)
            answer = self.generate(updated_query)
            if(self.check(query, answer)):
                break
            updated_query = updated_query + answer
        return answer

    def arag(self, prompt: str):
        complexity = self.get_class(prompt)
        if complexity == 0.0:
            return self.no_retrieve(prompt)
        elif complexity == 1.0:
            return self.single_retrieve(prompt)
        else:
            return self.multi_retrieve(prompt)