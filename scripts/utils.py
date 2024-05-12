import random
import os
import numpy as np
import torch
import json
import requests

def seed_environment(seed):
    """ 设置整个环境的随机种子 """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    return

def torch_gc(device):
    if torch.cuda.is_available():
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

def gen_response(query, history=[], max_length=None, top_p=None, temperature=None):
    # 如果想连续询问，应补充history，history是一个历史对话列表，列表中每个元素是一个长度为2的列表，由一问一答两个字符串组成
    url = "http://localhost:8000"
    payload = json.dumps({
        "query": query, 
        "history": history, 
        "max_length": max_length, 
        "top_p": top_p, 
        "temperature": temperature
    })
    headers = {'Content-Type': 'application/json'}

    full_response = requests.request("POST", url, headers=headers, data=payload)
    # 返回值response.json()包含 "response", "history", "status", "time", 例如：
    # {
    #     "response":"你好👋！我是人工智能助手 ChatGLM-6B，很高兴见到你，欢迎问我任何问题。",
    #     "history":[["你好","你好👋！我是人工智能助手 ChatGLM-6B，很高兴见到你，欢迎问我任何问题。"]],
    #     "status":200,
    #     "time":"2023-03-23 21:38:40"
    # }

    new_response = full_response.json()["response"]
    new_history = full_response.json()["history"]

    return new_response, new_history


if __name__ == '__main__':

    input = {
        'query': "123 + 123 + 123 + 123 = ？",
        'history': [['123 * 4 = ?', '123 * 4 = 496']],
    }
    request, _ = gen_response(**input)
    print(request)
