from fastapi import FastAPI
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

tokenizer = AutoTokenizer.from_pretrained('./pai-bloom-1b1-text2prompt-sd/')
model = AutoModelForCausalLM.from_pretrained('./pai-bloom-1b1-text2prompt-sd/').eval().cuda()
model.to(device)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/generate_prompt/")
async def generate_prompt(raw_prompt: str):
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
    env_file = ".env"
    logging.info("Loading environment from '%s'", env_file)
    load_dotenv(dotenv_path=env_file)
    host = os.getenv("API_HOST","0.0.0.0")
    port = int(os.getenv("API_PORT",7861))
    workers = int(os.getenv("API_WORKERS",4))

    uvicorn.run(app, host=host, port=port, workers=workers)
