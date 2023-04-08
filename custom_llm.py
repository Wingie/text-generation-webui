'''

This is an example on how to use the API for oobabooga/text-generation-webui.

Make sure to start the web UI with the following flags:

python server.py --model MODEL --listen --no-stream

Optionally, you can also add the --share flag to generate a public gradio URL,
allowing you to use the API remotely.

'''
import json

import requests

# Server address
server = "127.0.0.1"

# Generation parameters
# Reference: https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
params = {
    'max_new_tokens': 200,
    'do_sample': True,
    'temperature': 0.5,
    'top_p': 0.9,
    'typical_p': 1,
    'repetition_penalty': 1.05,
    'encoder_repetition_penalty': 1.0,
    'top_k': 0,
    'min_length': 0,
    'no_repeat_ngram_size': 0,
    'num_beams': 1,
    'penalty_alpha': 0,
    'length_penalty': 1,
    'early_stopping': False,
    'seed': -1,
}

# Input prompt


import torch
from langchain.llms.base import LLM
from llama_index import SimpleDirectoryReader, LangchainEmbedding, GPTListIndex, PromptHelper
from llama_index import LLMPredictor, ServiceContext
from transformers import pipeline
from typing import Optional, List, Mapping, Any


# define prompt helper
# set maximum input size
max_input_size = 2048
# set number of output tokens
num_output = 256
# set maximum chunk overlap
max_chunk_overlap = 20
prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)


class CustomLLM(LLM):
    model_name = "wingie/tbyw_v4" # 

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        payload = json.dumps([prompt, params])
        
        response = requests.post(f"http://{server}:7860/run/textgen", json={
            "data": [
                payload
            ]
        }).json()
        
        reply = response["data"][0]
        return response[prompt_length:]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name_of_model": self.model_name}

    @property
    def _llm_type(self) -> str:
        return "custom"

# define our LLM
llm_predictor = LLMPredictor(llm=CustomLLM())

service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

# Load the your data
documents = SimpleDirectoryReader('./training/datasets/').load_data()
index = GPTListIndex.from_documents(documents, service_context=service_context)

# Query and print response
response = new_index.query("what do you make of these documents?")
print(response)