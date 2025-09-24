#!/bin/bash

# 设置conda初始化
source ~/anaconda3/etc/profile.d/conda.sh 

# # 首次需创建conda环境
# conda create -n oasis python=3.11 -y

# 激活oasis环境
conda activate oasis

# 设置Python路径，确保可以找到本地oasis模块(未设置默认使用camel包)
export PYTHONPATH="/home/liying/code/oasis:$PYTHONPATH"

# 设置api_key，这里使用deepseek
export OPENAI_API_KEY="sk-71b28862eaa54af297d14a0766bbcf36"

# 运行Twitter模拟程序
python3 ./twitter_simulation/twitter_simulation.py --config ./twitter_simulation/twitter.yaml