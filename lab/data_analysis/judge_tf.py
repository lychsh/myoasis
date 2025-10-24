import requests
import json


API_URL = "https://api.deepseek.com/v1/chat/completions"
API_KEY = "" # your API key

def detect_fake_news(text):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    # DeepSeek API 使用 OpenAI 兼容格式
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "system",
                "content": "你是一个专业的虚假信息检测专家。请分析给定的文本内容，判断是否为虚假信息。请以JSON格式返回结果，包含is_fake（布尔值）和confidence（0-1之间的浮点数）字段。"
            },
            {
                "role": "user", 
                "content": f"请分析以下文本是否为虚假信息：\n\n{text}"
            }
        ],
        "temperature": 0.1,
        "max_tokens": 1000
    }

    try:
        response = requests.post(API_URL, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        
        # 解析返回的内容
        content = result['choices'][0]['message']['content']
        
        # 尝试解析JSON格式的回复
        try:
            parsed_result = json.loads(content)
            is_fake = parsed_result.get("is_fake", False)
            confidence = parsed_result.get("confidence", 0.5)
        except json.JSONDecodeError:
            if "虚假" in content or "假" in content or "不真实" in content:
                is_fake = True
            else:
                is_fake = False
                
        return is_fake, confidence
        
    except requests.exceptions.RequestException as e:
        print(f"API Wrong: {e}")
        return False, 0


news_text = '''
'''

if __name__ == '__main__':
    is_fake, confidence = detect_fake_news(news_text)
    if confidence > 0.5:
        print(f"Result:  {'Flase' if is_fake else 'True'}")
        print(f"Confidence: {confidence}")
