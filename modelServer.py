#!/usr/bin/env python
from fastapi import FastAPI
from langchain.schema.output_parser import StrOutputParser
from langserve import add_routes
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoTokenizer
import transformers
from transformers import AutoModelForCausalLM
import torch
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_use_double_quant=True,
    bnb_8bit_quant_type="nf8",
    bnb_8bit_compute_dtype=torch.bfloat16,
    llm_int8_enable_fp32_cpu_offload=True
)

model_id="google/gemma-7b-it"


model = AutoModelForCausalLM.from_pretrained(model_id,
                                             quantization_config=bnb_config,
                                             device_map="auto",
                                             trust_remote_code=True,
                                             token="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"
                                            )

tokenizer = AutoTokenizer.from_pretrained(model_id, token="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")

pipeline=transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=500
    )

llm=HuggingFacePipeline(pipeline=pipeline)

conversational_qa_chain = (
   llm | StrOutputParser()
)

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)
# Adds routes to the app for using the chain under:
# /invoke
# /batch
# /stream
add_routes(app, conversational_qa_chain)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)