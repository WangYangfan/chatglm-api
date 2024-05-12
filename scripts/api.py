from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModel
import accelerate
import uvicorn
import json
import datetime
import argparse
import os
from loguru import logger
from dotenv import load_dotenv

from utils import seed_environment, torch_gc


app = FastAPI()

@app.post("/")
async def create_item(request: Request):
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)

    query = json_post_list.get('query')
    history = json_post_list.get('history')
    max_length = json_post_list.get('max_length')
    top_p = json_post_list.get('top_p')
    temperature = json_post_list.get('temperature')

    response, history = model.chat(
        tokenizer=tokenizer,
        query=query,
        history=history,
        max_length=max_length if max_length else 2048,
        top_p=top_p if top_p else 0.7,
        temperature=temperature if temperature else 0.95,
    )

    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")

    answer = {
        'response': response,
        'history': history,
        'status': 200,
        'time': time,
    }

    torch_gc(device=device)
    
    return answer




if __name__ == '__main__':
    load_dotenv(dotenv_path="../envs/api.env", verbose=True, override=True)
    local_models = os.getenv('LOCAL_MDOELS')

    seed_environment(seed=42)
    accelerater = accelerate.Accelerator()
    device = accelerater.device
    logger.info("🥰 device: {}".format(device))

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--port', type=int, default=None)
    args = parser.parse_args()
    logger.info("🧩 model name: {}, port: {}".format(args.model_name, args.port))

    tokenizer = AutoTokenizer.from_pretrained(local_models + args.model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(local_models + args.model_name, trust_remote_code=True).float()

    model = accelerater.prepare(model)

    model.eval()
    uvicorn.run(app, host='localhost', port=args.port, workers=1)
    

