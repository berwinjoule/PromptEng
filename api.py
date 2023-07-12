from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import uvicorn
from dotenv import load_dotenv
import logging
import os


# 指定你想要使用的GPU设备，假设你有多个GPU设备，你想要在第一个设备上运行模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu") # CPU上运行模型，初步测试耗时约为GPU的5倍

app = FastAPI()
security = HTTPBasic()

env_file = ".env"
logging.info("Loading environment from '%s'", env_file)
load_dotenv(dotenv_path=env_file)

tokenizer = AutoTokenizer.from_pretrained('./pai-bloom-1b1-text2prompt-sd/')
model = AutoModelForCausalLM.from_pretrained('./pai-bloom-1b1-text2prompt-sd/').eval().cuda()
model.to(device)

def verify(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = os.getenv("API_USERNAME","admin")
    correct_password = os.getenv("API_PASSWD","passwd")
    if credentials.username != correct_username or credentials.password != correct_password:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    return True

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/generate_prompt/")
async def generate_prompt(raw_prompt: str, verified: bool = Depends(verify)):
    logging.info(f"raw_prompt: {raw_prompt}")
    input = f'Instruction: Give a simple description of the image to generate a drawing prompt.\nInput: {raw_prompt}\nOutput:'
    input_ids = tokenizer.encode(input, return_tensors='pt').to(device)

    outputs = model.generate(
        input_ids,
        max_length=384,
        do_sample=True,
        temperature=1.0,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2,
        num_return_sequences=1)

    prompts = tokenizer.batch_decode(outputs[:, input_ids.size(1):], skip_special_tokens=True)
    prompts = [p.strip() for p in prompts]
    return {"prompt": prompts[0]}

if __name__ == "__main__":
    host = os.getenv("API_HOST","0.0.0.0")
    port = int(os.getenv("API_PORT",7861))
    workers = int(os.getenv("API_WORKERS",1)) # workers=1，目前看work数大于1的话，会出现问题

    print(f"Starting server on {host}:{port} with {workers} workers")
    uvicorn.run(app, host=host, port=port, workers=workers)
