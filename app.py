import gradio as gr
import requests
import json
from diffusers import StableDiffusionPipeline
import torch


model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
if torch.cuda.is_available():
    pipe = pipe.to("cuda")


def inference(text):
    service_url = 'http://1502318844610933.cn-shanghai.pai-eas.aliyuncs.com/api/predict/bloom1b1_prompteng'
    datas = json.dumps([{
        "content": f"Instruction: Give a simple description of the image to generate a drawing prompt.\nInput: {text}.\nOutput:"
    }])
    head = {
        "Authorization": "N2ExMGE2ZjQ0YjMwMTYzYmFkZGJjZDgyZmQyNmQ1NTMxNGJmODNkMw=="
    }
    try:
        r = requests.post(service_url, data=datas, headers=head)
        generated_text = json.loads(r.text)[0]
    except:
        generated_text = text
    image1 = pipe(text).images[0]
    image2 = pipe(generated_text).images[0]
    return generated_text, image1, image2


demo = gr.Blocks()
with demo:
    with gr.Row():
        input_prompt = gr.Textbox(label="请输入您的prompt", 
                                    value="a bus",
                                    lines=3)
        b1 = gr.Button("开始生成")
        generated_txt = gr.Textbox(label="我们为您生成的prompt",
                                    lines=3)
    with gr.Row():
        image1 = gr.Image(label = '使用原prompt生成的图像')
        image2 = gr.Image(label = '使用我们生成的prompt生成的图像')
    b1.click(inference, inputs=[input_prompt], outputs=[generated_txt, image1, image2]) 
    
demo.launch(enable_queue=True)
