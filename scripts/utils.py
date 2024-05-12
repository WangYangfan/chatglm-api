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

def gen_response(query, history=[], max_length=None, top_p=None, temperature=None, port=None):
    """
    调用模型chat方法，输出模型回答和历史对话信息

    :param history: 如果连续询问，应补充history参数，是一个历史对话列表
        if chatglm-6b or chatglm2-6b:
            列表组成的列表，列表中每个元素是一个长度为2的列表，由一问一答两个字符串组成
        if chatglm3-6b:
            字典组成的列表，列表中每个元素包括键role和context，由user和assistant交替组成

    :var full_response: full_response.json()包含 "response", "history", "status", "time", 例如：
        {
            "response":"你好👋！我是人工智能助手 ChatGLM-6B，很高兴见到你，欢迎问我任何问题。",
            "history":[["你好","你好👋！我是人工智能助手 ChatGLM-6B，很高兴见到你，欢迎问我任何问题。"]],
            "status":200,
            "time":"2023-03-23 21:38:40"
        }

    :return new_response, new_history: 模型回答和历史信息
    """

    url = "http://localhost:" + str(port)
    payload = json.dumps({
        "query": query, 
        "history": history, 
        "max_length": max_length, 
        "top_p": top_p, 
        "temperature": temperature
    })
    headers = {'Content-Type': 'application/json'}

    full_response = requests.request("POST", url, headers=headers, data=payload)



    new_response = full_response.json()["response"]
    new_history = full_response.json()["history"]

    return new_response, new_history


if __name__ == '__main__':

    input = {
        'query': "123 + 123 + 123 + 123 = ？",
        # 'history': [['123 * 4 = ?', '123 * 4 = 496']],  # if chatglm-6b or chatglm2-6b
        # 'history': [{'role': 'user', 'content': '123 * 4 = ?'}, {'role': 'assistant', 'metadata': '', 'content': '123 * 4 = 496'}], # if chatglm3-6b
        'port': 8000,   # or 8001, 8002
    }
    request, history = gen_response(**input)
    print(request, history)
