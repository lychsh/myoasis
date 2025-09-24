import os
from transformers import AutoTokenizer, AutoModel

# 设置镜像源
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

model_name = "Twitter/twhin-bert-base"
cache_dir = "/path/to/your/custom/directory"  # 指定自定义路径

# 下载到指定目录
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)

print(f"模型已下载到: {cache_dir}")