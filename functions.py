import torch
from llama_index.core import StorageContext, Settings
from llama_index.llms.huggingface import HuggingFaceLLM # type: ignore
from transformers import BitsAndBytesConfig
from llama_index.core.prompts import PromptTemplate
from llama_index.retrievers.bm25 import BM25Retriever # type: ignore
import logging
import sys
import utils
import time
from PyPDF2 import PdfReader
import requests
import re

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# variables to be filles in the chat
query = ''
reranked_nodes = []

# convert data from bytes to text, only for processing PDF files
def processPdfFile(pdf_docs):
    """ format bytes -> text for pdf files """
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# streaming functionality in chat window
def textToGenerator(text):
    for t in text.split():
        yield t + ' '
        time.sleep(0.10)

# BM25 retriever
def retrieveNodes(query, nodes, k):
    retriever = BM25Retriever.from_defaults(nodes = nodes, similarity_top_k=k)
    nodes = retriever.retrieve(query) 
    return nodes if nodes else []

# re-ranker, built in get_score gives the similarity of the nodes with the query
def rerankNodes(nodes, k):
    return sorted(nodes, key=lambda x: x.get_score(), reverse=True)[:k]

# prompts
SYSTEM_PROMPT = """You are a helptul AI assistant, you are tasked to do either one of the two tasks which will be specified in the query given.
Task1 is to generate an answer to the query using the context in step by step method similar to Chain of Thought.
Task2 is to generate a concise and comprehensive answer to the query using the contents of the context retrieved from documents or a codebase.
Perform task1 or task2 depending on the type specified in the query """

def chainOfThoughtPrompt(reranked_nodes, query):
    task1_prompt = f"""task1: "Chain of Thought answer" - Use the context from the documents retrieved to give a step by step answer to the query specified.
    Focus on incorporating key terms, synonyms, related concepts, and descriptive phrases to enhance the answer's scope and accuracy.
    Example:
    context: {reranked_nodes},
    query: {query},
    answer: Let's think step by step.... generate the answer step by step... in a maximum of 5-6 steps"""
    return task1_prompt

def QAGenerationPrompt(reranked_nodes, query):
    task2_prompt = f"""task2: "Answer using context" - You are given context: {reranked_nodes} retrieved using the query: {query}.
    You have to to use the content provided in the context and output a concise and comprehensive answer to the query, You have to output
    only your answer without the context and the query. """
    return task2_prompt

def cleanResponce(task, query, responce):
    return responce.replace(task, ' ').replace(query, ' ')

# HUGGING FACE inference
def getResponse(input):
    API_URL = f"https://api-inference.huggingface.co/models/{utils.LLM_NAME}"
    headers = {"Authorization": "Bearer xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"} # hugging face access token

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()
        
    output = query({
        "inputs": f"{input}",
        "parameters": {'max_new_tokens': utils.MAX_NEW_TOKENS}
    })
    s = output[0]['generated_text']
    s = s.replace('\n', ' ')
    return output[0]['generated_text'][-1050:]
