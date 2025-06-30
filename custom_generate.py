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

def generate_until_pattern(model, tokenizer, initial_prompt, pattern, max_length=2048):
    # Get the EOS token ID
    eos_token_id = tokenizer.eos_token_id
    
    # Encode initial prompt
    input_ids = tokenizer.encode(initial_prompt, return_tensors='pt').to(model.device)
    
    output_tokens = copy.deepcopy(input_ids)
    # Create attention mask
    attention_mask = torch.ones_like(input_ids)
    
    # Prepare past key values (KV cache)
    past_key_values = None
    
    # Keep track of just the generated text separately
    generated_text = ""
    current_length = input_ids.shape[1]
    
    while current_length < max_length:
        # Generate next token with KV cache
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                past_key_values=past_key_values
            )
            
            # Get logits and past key values
            logits = outputs.logits
            past_key_values = outputs.past_key_values
            
            # Get the last token's prediction
            next_token_logits = logits[0, -1, :]
            next_token_id = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)
            output_tokens = torch.cat([output_tokens, next_token_id], dim = -1)
            # Check for EOS token
            if next_token_id.item() == eos_token_id:
                return tokenizer.decode(output_tokens[0], skip_special_tokens = True), False, None
            
            # Decode the token
            next_token_text = tokenizer.decode(
                next_token_id[0],
                skip_special_tokens=True
            )
            generated_text += next_token_text
            for match in re.finditer(pattern, generated_text, re.DOTALL):
                return tokenizer.decode(output_tokens[0], skip_special_tokens = True), True, (match.start(), match.end())
            # Update input_ids for next iteration
            input_ids = next_token_id
            attention_mask = torch.ones_like(input_ids)
            
            current_length += 1
    return tokenizer.decode(output_tokens[0], skip_special_tokens = True), False, None  # Return if max_length reached