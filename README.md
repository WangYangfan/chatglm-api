# chatglm-api

api deployment of  `chatglm-6b, chatglm2-6b, chatglm3-6b` and calling by python scripts.

**Requirements**
```
transformers==4.40.2
accelerate==0.30.0
torch==2.3.0
fastapi
argparse
loguru
dotenv
```

## Setup

Before run scripts, need setup env: in `envs/api.env`, `LOCAL_MODELS` should be local root path of models. If using remote models, just keep `LOCAL_MODELS=""`.

## Hang Models

**Hang one model**

```bash
CUDA_VISIBLE_DEVICES=1 python api.py --model_name THUDM/chatglm-6b --port 8000
```

which means that a model named `THUDM/chatglm-6b` is hung on `cuda:1` with port `8000`.

**Hang multi model**

```bash
CUDA_VISIBLE_DEVICES=1 python api.py --model_name THUDM/chatglm-6b --port 8000

CUDA_VISIBLE_DEVICES=2 python api.py --model_name THUDM/chatglm2-6b --port 8001

CUDA_VISIBLE_DEVICES=3 python api.py --model_name THUDM/chatglm3-6b --port 8002
```

the script `api.sh` is suitable to all of `chatglm-6b, chatglm2-6b, chatglm3-6b`.

**Calling**

```python
input = {
    'query': "123 + 123 + 123 + 123 = ？",
    # 'history': [['123 * 4 = ?', '123 * 4 = 496']],  # if chatglm-6b or chatglm2-6b
    # 'history': [{'role': 'user', 'content': '123 * 4 = ?'}, {'role': 'assistant', 'metadata': '', 'content': '123 * 4 = 496'}], # if chatglm3-6b
    'port': 8000,   # or 8001, 8002
}
request, history = gen_response(**input)
print(request, history)
```

**Attention: `chatglm-6b, chatglm2-6b` and `chatglm3-6b` have different `history` forms when called.**

